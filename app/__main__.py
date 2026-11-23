import argparse
import asyncio
import logging

from aiogram import Dispatcher

from app.agent import AgentService
from app.agent_router import AgentRouter
from app.briefing import BriefingBuilder
from app.config import ConfigError, Settings
from app.db import create_pool
from app.delivery import DeliveryService
from app.handlers.assistant import create_router
from app.health import HealthState, start_health_server
from app.llm import OpenRouterChatClient
from app.logging_setup import configure_logging
from app.mcp_tools.client import MCPClient
from app.notifications import NotificationService
from app.service import AssistantService
from app.store import PostgresStore
from app.supervisor import SpecialistRouter
from app.telegram import create_bot
from app.tracing import TraceRecorder

logger = logging.getLogger("app")


async def run(settings: Settings) -> None:
    state = HealthState()
    bot = create_bot(settings)
    runner = None
    try:
        state.pool = await create_pool(settings)
        logger.info("PostgreSQL подключён: SELECT 1 выполнен.")
        # Начальная проверка токена и маршрута через прокси ограничена по времени.
        async with asyncio.timeout(30):
            await bot.get_me()
            webhook = await bot.get_webhook_info()
        if webhook.url:
            raise ConfigError(
                "У бота установлен webhook. Удалите его перед запуском polling "
                "или используйте отдельного учебного бота."
            )
        logger.info("Telegram доступен. Запускается polling.")
        store = PostgresStore(state.pool)
        await store.initialize()
        async with OpenRouterChatClient(settings) as llm, MCPClient(settings) as mcp:
            agent = AgentService(
                store,
                AgentRouter(llm),
                mcp,
                settings,
                specialist_router=SpecialistRouter(llm),
                traces=TraceRecorder(state.pool),
            )
            dispatcher = Dispatcher()
            dispatcher.include_router(
                create_router(
                    AssistantService(store, llm, settings, agent),
                    notifications=NotificationService(state.pool),
                )
            )
            runner = await start_health_server(state, settings.health_port)
            state.polling_task = asyncio.create_task(
                dispatcher.start_polling(
                    bot,
                    handle_as_tasks=False,
                    allowed_updates=dispatcher.resolve_used_update_types(),
                    close_bot_session=False,
                )
            )
            delivery_task = asyncio.create_task(
                DeliveryService(state.pool, bot, briefing_builder=BriefingBuilder(mcp)).run()
            )
            state.initialized = True
            try:
                await state.polling_task
            finally:
                delivery_task.cancel()
                await asyncio.gather(delivery_task, return_exceptions=True)
    finally:
        state.initialized = False
        if state.polling_task and not state.polling_task.done():
            state.polling_task.cancel()
            await asyncio.gather(state.polling_task, return_exceptions=True)
        if runner:
            await runner.cleanup()
        await bot.session.close()
        if state.pool:
            try:
                async with asyncio.timeout(10):
                    await state.pool.close()
            except TimeoutError:
                state.pool.terminate()


def main() -> int:
    parser = argparse.ArgumentParser(description="Учебный AI-ассистент в Telegram")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    try:
        settings = Settings.load(args.env_file)
    except ConfigError as exc:
        print(str(exc))
        return 1
    configure_logging(settings)
    try:
        asyncio.run(run(settings))
    except KeyboardInterrupt:
        logger.info("Бот остановлен.")
    except Exception:
        logger.exception("Не удалось запустить бот. Проверьте БД, токен и прокси.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
