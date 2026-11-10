"""Общие фикстуры тестов, учитывающие запрет локальных сокетов в песочнице."""

import socket

import pytest


@pytest.fixture
def unused_tcp_port():
    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", 0))
        except PermissionError:
            pytest.skip("Среда запрещает bind(127.0.0.1); нужен локальный сетевой сокет")
        return listener.getsockname()[1]
