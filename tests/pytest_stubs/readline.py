"""No-op test-only replacement for the broken Python 3.13 libedit module.

Pytest imports :mod:`readline` only as an early stdio workaround.  The local
extension currently segfaults during module creation on this macOS runtime;
none of this repository's tests require interactive line editing.
"""


def _noop(*_args, **_kwargs):
    return None


set_completer = _noop
set_completer_delims = _noop
parse_and_bind = _noop
insert_text = _noop
redisplay = _noop


def __getattr__(_name):
    return _noop
