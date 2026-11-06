"""Detect training started from an unguarded script before spawn workers re-run it.

Torch-free so lightweight installs and tests can import it.
"""
from pathlib import Path
import ast
import sys

from .run_controller import FatalExecutionError


def _is_main_check(test):
    """``__name__ == "__main__"``, alone or as one operand of an ``and``."""
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        return any(_is_main_check(value) for value in test.values)
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)):
        return False
    sides = [test.left, *test.comparators]
    return (any(isinstance(side, ast.Name) and side.id == '__name__' for side in sides)
            and any(isinstance(side, ast.Constant) and side.value == '__main__' for side in sides))


def _main_guard_lines(path):
    """Line ranges of every ``if __name__ == "__main__":`` body in a script."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'), filename=str(path))
    return [(node.body[0].lineno, node.body[-1].end_lineno) for node in ast.walk(tree)
            if isinstance(node, ast.If) and _is_main_check(node.test)]


def unguarded_main_call():
    """``(path, line)`` when this call came from a script's unguarded top level, else None.

    Previews and evaluations run in ``spawn`` worker processes. Spawn re-imports
    a script run as ``python script.py`` (or ``-m module``) in every worker, so
    module-level code outside the main guard, such as ``hg.train`` itself or an
    ``rmtree`` before it, would run again inside each worker.
    """
    main = sys.modules.get('__main__')
    spec_name = getattr(getattr(main, '__spec__', None), 'name', None)
    if spec_name is not None and (spec_name == '__main__' or spec_name.endswith('.__main__')):
        return None  # spawn does not re-run a package's __main__ (python -m pkg).
    path = getattr(main, '__file__', None)
    if main is None or path is None:
        return None  # Interactive sessions and ``python -c`` have nothing to re-run.
    frame, lines = sys._getframe(1), []
    while frame is not None:
        if frame.f_globals is main.__dict__ and frame.f_code.co_name == '<module>':
            lines.append(frame.f_lineno)
        frame = frame.f_back
    if not lines:
        return None
    try:
        guarded = _main_guard_lines(path)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    for line in lines:
        if not any(start <= line <= end for start, end in guarded):
            return path, line
    return None


def require_main_guard(purpose):
    """Fail before any worker is spawned instead of re-running the caller's script."""
    caller = unguarded_main_call()
    if caller is not None:
        path, line = caller
        raise FatalExecutionError(
            f'{purpose} run in spawned worker processes, which re-import the main script. '
            f'{path}:{line} starts training from module level without a main guard, so every '
            'worker would run that script again. Move the training code under '
            '`if __name__ == "__main__":`.')
