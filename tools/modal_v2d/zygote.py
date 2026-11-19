"""Resident process: load a module call's dependencies before the call, then run it once.

    python zygote.py <module>        (cwd = the image workdir, env = the call's env)

Runs under the module image's own Python, stdlib only (also used by tools/fastrecon).

1. Preload: execute every top-level import statement of <module> (the heavy library
   imports), then call `preload(argv_hint)` from `<module package>._resident` if the
   image has one; that hook builds model objects and registers them with
   `v2d.common.resident.put`. Prints READY_MARKER when done.
2. Read one JSON line from stdin: {"argv": [...]} = the exact `python -m <module> ...`
   command of the real call (argv_hint comes from the V2D_ZYGOTE_ARGV_HINT env var).
3. Run it in this process exactly as `python -m` would (runpy, run_name="__main__") and
   exit with its exit code. One call per process: nothing carries over to another call.

The caller (worker / replay_runner) guarantees that the call's env and workdir equal
the ones this process was started with; otherwise it does not use the zygote.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
import runpy
import sys
import time
import traceback

READY_MARKER = "\x1e[v2d-zygote] READY"
MISMATCH_EXIT = 97


def _say(text: str) -> None:
    print(f"[v2d-zygote] {text}", flush=True)


def _preimport(module: str) -> int:
    """Execute the module's top-level imports (not its body); return how many ran."""
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return 0
    with open(spec.origin, encoding="utf-8") as stream:
        tree = ast.parse(stream.read(), spec.origin)
    package = module.rpartition(".")[0]
    count = 0
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        try:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    importlib.import_module(alias.name)
            else:
                base = node.module or ""
                if node.level:
                    base = importlib.util.resolve_name("." * node.level + base, package)
                mod = importlib.import_module(base)
                for alias in node.names:  # `from x import submodule`
                    if alias.name != "*" and not hasattr(mod, alias.name):
                        try:
                            importlib.import_module(f"{base}.{alias.name}")
                        except ImportError:
                            pass
            count += 1
        except Exception as exc:  # noqa: BLE001  - the real run will import (and fail) itself
            _say(f"pre-import skipped ({type(exc).__name__}: {exc})")
    return count


def main() -> int:
    module = sys.argv[1]
    # `python -m` puts the working directory first on sys.path; this script's own
    # directory (tools/modal_v2d) must not shadow anything the module imports.
    sys.path[0] = os.getcwd()
    t0 = time.time()
    hint = json.loads(os.environ.get("V2D_ZYGOTE_ARGV_HINT", "[]"))
    try:
        imported = _preimport(module)
        t1 = time.time()
        hooked = False
        try:
            hook = importlib.import_module(module.rpartition(".")[0] + "._resident")
        except ModuleNotFoundError as exc:
            if not exc.name or not exc.name.endswith("._resident"):
                raise
            hook = None
        if hook is not None and hasattr(hook, "preload"):
            hook.preload(module, hint)
            hooked = True
        _say(f"preloaded {module}: {imported} imports {t1 - t0:.1f}s"
             + (f", preload hook {time.time() - t1:.1f}s" if hooked else ""))
    except Exception:  # noqa: BLE001  - a failed preload just means a cold run
        _say("preload failed; the call will load everything itself:\n" + traceback.format_exc())
    print(READY_MARKER, flush=True)

    line = sys.stdin.readline()
    if not line:
        _say("no call arrived; exiting")
        return 0
    argv = json.loads(line)["argv"]
    if len(argv) < 3 or argv[1] != "-m" or argv[2] != module:
        _say(f"call {argv[:3]} does not match the preloaded module {module}")
        return MISMATCH_EXIT
    sys.argv = [module] + argv[3:]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr, flush=True)
        return 1
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
