"""Slow live control files cannot stall update-thread polling."""
from threading import Event, Thread

from hypergan.background_poll import BackgroundPoll


def test_one_read_in_flight_and_close_does_not_wait_for_blocked_io():
    entered, release = Event(), Event()
    calls = []
    def read():
        calls.append(True)
        entered.set()
        assert release.wait(5)
        return {'result': 1}
    reader = BackgroundPoll(read, interval=0)
    try:
        assert reader.poll() == (False, None)
        assert entered.wait(5)
        for _ in range(1000):
            assert reader.poll() == (False, None)
        assert len(calls) == 1
        reader.close()
        assert reader.thread.is_alive()
    finally:
        release.set()
        reader.close(wait=True)
    assert not reader.thread.is_alive()


def test_explicit_final_read_waits_then_refreshes_stale_cached_result():
    entered, release, finished = Event(), Event(), Event()
    calls = []
    def read():
        calls.append(True)
        value = len(calls)
        if value == 1:
            entered.set()
            assert release.wait(5)
        return value
    reader = BackgroundPoll(read)
    outcome = []
    def final_read():
        outcome.append(reader.read_now())
        finished.set()
    try:
        reader.poll()
        assert entered.wait(5)
        final = Thread(target=final_read)
        final.start()
        assert not finished.wait(.05)
        release.set()
        assert finished.wait(5)
        final.join()
        assert outcome == [2]
    finally:
        release.set()
        reader.close(wait=True)
