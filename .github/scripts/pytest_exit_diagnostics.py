"""Keep teardown diagnostics active after pytest has removed its own timer."""
import faulthandler
import json
import multiprocessing
import os
import sys
import threading

import pytest


def main():
    result = int(pytest.main(sys.argv[1:]))
    report = {
        'event': 'pytest_returned', 'pid': os.getpid(), 'exit_code': result,
        'children': [{'pid': child.pid, 'name': child.name, 'exit_code': child.exitcode,
                      'alive': child.is_alive()}
                     for child in multiprocessing.active_children()],
        'threads': [{'name': thread.name, 'daemon': thread.daemon, 'alive': thread.is_alive()}
                    for thread in threading.enumerate()],
    }
    print(json.dumps(report, sort_keys=True), file=sys.stderr, flush=True)
    faulthandler.enable(all_threads=True)
    faulthandler.dump_traceback_later(30, repeat=True)
    return result


if __name__ == '__main__':
    # Preserve pytest's result and normal interpreter cleanup. A stuck child,
    # thread or native finalizer must still fail the bounded CI job.
    raise SystemExit(main())
