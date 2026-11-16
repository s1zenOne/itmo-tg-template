"""Общие фикстуры тестов, учитывающие запрет локальных сокетов в песочнице."""

import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import pytest

from app.config import Settings


@pytest.fixture
def unused_tcp_port():
    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", 0))
        except PermissionError:
            pytest.skip("Среда запрещает bind(127.0.0.1); нужен локальный сетевой сокет")
        return listener.getsockname()[1]


@pytest.fixture
def lab2_database():
    """Изолированный временный PostgreSQL на приватном Unix-сокете."""
    if os.environ.get("RUN_LAB2_POSTGRES") != "1":
        pytest.skip("Включите RUN_LAB2_POSTGRES=1 для проверки на PostgreSQL")
    initdb = shutil.which("initdb")
    pg_ctl = shutil.which("pg_ctl")
    if not initdb or not pg_ctl or os.name != "posix" or os.getuid() == 0:
        pytest.fail("Для RUN_LAB2_POSTGRES=1 нужны initdb и pg_ctl от обычного пользователя")

    # Короткий путь на macOS укладывается в ограничение длины Unix-сокета.
    with tempfile.TemporaryDirectory(prefix="itmo-lab2-pg-", dir="/tmp") as temporary:
        root = Path(temporary)
        data = str(root / "data")
        subprocess.run(
            [initdb, "-D", data, "-A", "trust", "-U", "bot", "--no-locale", "-E", "UTF8"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        try:
            subprocess.run(
                [
                    pg_ctl,
                    "-D",
                    data,
                    "-l",
                    str(root / "postgres.log"),
                    "-w",
                    "-t",
                    "20",
                    "-o",
                    f"-c listen_addresses='' -k {temporary}",
                    "start",
                ],
                check=True,
                capture_output=True,
                timeout=25,
            )
            yield Settings(
                bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
                postgres_password="unused-local-socket",
                postgres_host=temporary,
                postgres_db="postgres",
            )
        finally:
            if Path(data, "postmaster.pid").exists():
                subprocess.run(
                    [pg_ctl, "-D", data, "-w", "-t", "20", "-m", "immediate", "stop"],
                    check=True,
                    capture_output=True,
                    timeout=25,
                )
