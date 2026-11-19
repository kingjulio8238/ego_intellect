"""Hand a model loaded ahead of time to the call that needs it (resident execution).

A resident process (tools/modal_v2d/zygote.py) starts before its call is due, imports
the entry module's dependencies and runs the image's optional preload hook, which builds
the expensive objects (model weights on the GPU) and registers them here with `put`.
When the call arrives it runs in that same process, and the loader asks `take` first.

Outside a resident process nothing is ever registered, so `take` returns None and every
loader builds its object exactly as before. A key carries every argument the object was
built from, so a call with different arguments never receives it (it builds its own and
the preloaded objects are dropped first, so they never share GPU memory).
"""

from __future__ import annotations

from typing import Any, Hashable

_CACHE: dict[Hashable, Any] = {}


def put(key: Hashable, value: Any) -> None:
    _CACHE[key] = value


def take(key: Hashable) -> Any | None:
    """The object preloaded under `key`, handed over once; None if there is none."""
    if key in _CACHE:
        return _CACHE.pop(key)
    if _CACHE:  # preloaded for other arguments: free it before the caller builds its own
        _CACHE.clear()
        try:
            import gc  # noqa: PLC0415

            gc.collect()
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    return None


def keys() -> list[Hashable]:
    return list(_CACHE)
