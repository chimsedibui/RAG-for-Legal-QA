"""Streaming workers stop at the next pipeline yield after a disconnect."""
import importlib.util
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.config import Settings


@pytest.fixture
def app_module(monkeypatch, required_env):
    monkeypatch.setattr('core.config.get_settings', lambda: Settings())
    monkeypatch.setattr('providers.faiss_store.FaissVectorStore', lambda *args: object())
    monkeypatch.setattr('services.search.load_search_data', lambda *args: ({}, {}, {}))
    spec = importlib.util.spec_from_file_location(
        'cancellation_test_app', Path(__file__).resolve().parents[1] / 'api/app.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cancelled_worker_does_not_start_pipeline(app_module):
    cancelled = threading.Event()
    cancelled.set()
    calls = []

    def process(**kwargs):
        calls.append('started')
        yield {}

    app_module.pipeline = SimpleNamespace(process=process)
    output = queue.Queue()
    app_module._run_pipeline_in_thread([], True, 'off', output, cancelled)
    assert calls == []
    assert output.get_nowait() is app_module._SENTINEL


def test_disconnect_closes_pipeline_without_forwarding_stale_event(app_module):
    cancelled = threading.Event()
    closed = []

    def process(**kwargs):
        try:
            yield {'first': True}
            cancelled.set()
            yield {'stale': True}
            pytest.fail('Pipeline continued after cancellation')
        finally:
            closed.append(True)

    app_module.pipeline = SimpleNamespace(process=process)
    output = queue.Queue()
    app_module._run_pipeline_in_thread([], True, 'off', output, cancelled)
    assert output.get_nowait() == {'first': True}
    assert output.get_nowait() is app_module._SENTINEL
    assert output.empty()
    assert closed == [True]
