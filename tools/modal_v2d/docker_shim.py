"""A `docker` CLI stand-in for Modal role containers (stdlib only).

The repository drives every model through `docker run`/`docker exec` (reconstruction
host wrappers, weight downloads, `robotic_grounding/workflow/run.sh e2e-run`,
`groot_finetune/closed_loop/run_eval.sh`). Modal containers cannot run a docker daemon,
so this shim implements exactly the subset those scripts use:

* an image that *is* the current container (``V2D_MODAL_LOCAL_IMAGE``) runs in-process,
  with bind mounts emulated by symlinks;
* any other image becomes a Modal function call, dispatched through the parent
  process's RPC socket (the parent syncs bind-mounted directories via the state volume);
* `docker build` is a no-op for images the Modal app defines (they are built by Modal).

Anything outside that subset exits non-zero with an explicit message instead of
pretending to succeed.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.modal_v2d import config as C  # noqa: E402
from tools.modal_v2d import containerenv, mounts  # noqa: E402

STATE_DIR = Path(os.environ.get("V2D_SHIM_STATE", C.SHIM_STATE)) / "containers"

RUN_FLAGS = {"--rm", "-d", "--detach", "-i", "--interactive", "-t", "--tty", "--init", "--privileged"}
RUN_OPTIONS = {
    "--runtime", "--gpus", "--network", "--net", "--name", "--user", "-u", "-v", "--volume",
    "-e", "--env", "--entrypoint", "--group-add", "-w", "--workdir", "--shm-size", "--ipc",
    "--ulimit", "--cap-add", "--security-opt", "--hostname", "--platform", "--pull",
}
UNSUPPORTED_RUN_OPTIONS = {"--mount", "--env-file", "-p", "--publish", "--device", "--volumes-from"}
EXEC_FLAGS = {"-i", "--interactive", "-t", "--tty"}
EXEC_OPTIONS = {"-w", "--workdir", "-e", "--env", "-u", "--user"}


class ShimError(RuntimeError):
    pass


def _fail(message: str, code: int = 2) -> int:
    print(f"docker (v2d Modal shim): {message}", file=sys.stderr)
    return code


def normalize_image(image: str) -> str:
    name = image.rsplit("/", 1)[-1]
    return name.split(":", 1)[0]


def known_images() -> set[str]:
    raw = os.environ.get("V2D_MODAL_IMAGES")
    return set(raw.split(",")) if raw else set(C.ALL_IMAGES)


def local_image() -> str:
    return os.environ.get("V2D_MODAL_LOCAL_IMAGE", "")


# --------------------------------------------------------------------------- parsing
def _split_short_flags(token: str, flags: set[str]) -> list[str] | None:
    if re.fullmatch(r"-[a-zA-Z]{2,}", token) and all(f"-{ch}" in flags for ch in token[1:]):
        return [f"-{ch}" for ch in token[1:]]
    return None


def parse_options(
    args: list[str], flags: set[str], options: set[str], unsupported: set[str] = frozenset()
) -> tuple[dict[str, list[str]], set[str], list[str]]:
    """Parse leading docker options; return (option values, flags, remaining args)."""
    values: dict[str, list[str]] = {}
    seen: set[str] = set()
    index = 0
    while index < len(args):
        token = args[index]
        if not token.startswith("-") or token == "-":
            break
        if token == "--":
            index += 1
            break
        split = _split_short_flags(token, flags)
        if split:
            seen.update(split)
            index += 1
            continue
        name, has_value, value = token.partition("=")
        if name in unsupported:
            raise ShimError(f"option {name} is not supported")
        if name in flags and not has_value:
            seen.add(name)
            index += 1
            continue
        if name in options:
            if not has_value:
                if index + 1 >= len(args):
                    raise ShimError(f"option {name} requires a value")
                value = args[index + 1]
                index += 1
            values.setdefault(name, []).append(value)
            index += 1
            continue
        raise ShimError(f"unrecognised option {token!r}")
    return values, seen, args[index:]


def parse_volume(spec: str) -> dict[str, str]:
    parts = spec.split(":")
    if len(parts) == 1 and spec.startswith("/"):
        # Anonymous volume (e.g. FoundationPose's DEV_PRESERVE_VOLUMES): it only preserves
        # image content under a --dev bind mount, which the Modal executor never shadows.
        return {"src": "", "dst": spec, "mode": "anonymous"}
    if len(parts) < 2 or not parts[0].startswith("/"):
        raise ShimError(f"only absolute host bind mounts are supported, got -v {spec!r}")
    src, dst = parts[0], parts[1]
    mode = parts[2] if len(parts) > 2 else "rw"
    return {"src": os.path.realpath(src), "dst": dst, "mode": mode}


def parse_env(entries: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for entry in entries:
        key, has_value, value = entry.partition("=")
        if has_value:
            env[key] = value
        elif key in os.environ:  # `-e NAME` inherits from the caller, like docker
            env[key] = os.environ[key]
    return env


# --------------------------------------------------------------------------- state
def _state_path(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ShimError(f"invalid container name {name!r}")
    return STATE_DIR / f"{name}.json"


def load_container(name: str) -> dict | None:
    path = _state_path(name)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_container(record: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_path(record["name"]).write_text(json.dumps(record, indent=2))


def remove_container(name: str) -> bool:
    record = load_container(name)
    if record is None:
        return False
    mounts.revert(record.get("applied", []))
    _state_path(name).unlink()
    return True


def all_containers() -> list[dict]:
    if not STATE_DIR.exists():
        return []
    return [json.loads(path.read_text()) for path in sorted(STATE_DIR.glob("*.json"))]


# --------------------------------------------------------------------------- execution
def run_local(argv: list[str], *, image: str, env: dict[str, str], workdir: str | None) -> int:
    if not argv:
        raise ShimError("no command to run")
    cwd = workdir or containerenv.image_workdir()
    return subprocess.call(argv, cwd=cwd, env=containerenv.container_env(env, image))


def run_remote(spec: dict) -> int:
    """Send one container spec to the parent process and stream its log."""
    path = os.environ.get("V2D_MODAL_RPC", C.RPC_SOCKET)
    if not os.path.exists(path):
        raise ShimError(
            f"no Modal RPC socket at {path}; image {spec['image']!r} can only run inside a "
            "role container started by tools/modal_v2d/app.py"
        )
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(path)
        request = {"op": "run", "spec": spec, "caller": os.environ.get("V2D_MODAL_CALLER")}
        client.sendall((json.dumps(request) + "\n").encode())
        reader = client.makefile("r", encoding="utf-8")
        for line in reader:
            message = json.loads(line)
            if "log" in message:
                sys.stdout.write(message["log"])
                sys.stdout.flush()
            elif "exit" in message:
                return int(message["exit"])
            elif "error" in message:
                raise ShimError(message["error"])
    raise ShimError("Modal RPC connection closed without an exit status")


def _container_argv(entrypoint: list[str] | None, command: list[str]) -> list[str]:
    return [*(entrypoint or []), *command]


def cmd_run(args: list[str]) -> int:
    values, flags, rest = parse_options(args, RUN_FLAGS, RUN_OPTIONS, UNSUPPORTED_RUN_OPTIONS)
    if not rest:
        raise ShimError("docker run requires an image")
    image, command = normalize_image(rest[0]), rest[1:]
    if image not in known_images():
        raise ShimError(f"no Modal image is defined for {rest[0]!r} (known: {sorted(known_images())})")
    volumes = [parse_volume(spec) for spec in values.get("-v", []) + values.get("--volume", [])]
    env = parse_env(values.get("-e", []) + values.get("--env", []))
    gpu = bool(values.get("--gpus")) or "nvidia" in values.get("--runtime", [])
    entrypoint = values.get("--entrypoint")
    workdir = (values.get("-w") or values.get("--workdir") or [None])[-1]
    name = (values.get("--name") or [None])[-1]
    detached = "-d" in flags or "--detach" in flags
    is_local = image == local_image()

    if detached:
        if not name:
            raise ShimError("detached containers must be named")
        if load_container(name):
            raise ShimError(f"container {name} already exists")
        record = {
            "name": name,
            "image": image,
            "mounts": volumes,
            "env": env,
            "gpu": gpu,
            "network": (values.get("--network") or values.get("--net") or ["bridge"])[-1],
            "workdir": workdir,
            "local": is_local,
            "applied": [],
        }
        if is_local:
            record["applied"] = mounts.apply(volumes)
        save_container(record)
        print(name)
        return 0

    argv = _container_argv(entrypoint, command)
    if is_local:
        applied = mounts.apply(volumes)
        try:
            return run_local(argv, image=image, env=env, workdir=workdir)
        finally:
            mounts.revert(applied)
    return run_remote(
        {"image": image, "argv": argv, "env": env, "workdir": workdir, "mounts": volumes, "gpu": gpu}
    )


def cmd_exec(args: list[str]) -> int:
    values, _, rest = parse_options(args, EXEC_FLAGS, EXEC_OPTIONS)
    if len(rest) < 2:
        raise ShimError("docker exec requires a container and a command")
    name, command = rest[0], rest[1:]
    record = load_container(name)
    if record is None:
        return _fail(f"Error: No such container: {name}", 1)
    env = {**record["env"], **parse_env(values.get("-e", []) + values.get("--env", []))}
    workdir = (values.get("-w") or values.get("--workdir") or [record.get("workdir")])[-1]
    if record["local"]:
        return run_local(command, image=record["image"], env=env, workdir=workdir)
    if command in (["true"], [":"]):
        # Liveness probe (`setup` starts the E2E container with `docker exec <c> true`).
        # The recorded container is "running" by construction; renting a GPU container
        # to run `true` would only cost a cold start.
        return 0
    return run_remote(
        {
            "image": record["image"],
            "argv": command,
            "env": env,
            "workdir": workdir,
            "mounts": record["mounts"],
            "gpu": record["gpu"],
        }
    )


_MOUNTS_RANGE = re.compile(r"^\{\{range \.Mounts\}\}(.*)\{\{end\}\}$", re.S)


def render_inspect(record: dict, template: str) -> str:
    match = _MOUNTS_RANGE.match(template)
    if match:
        inner = match.group(1)
        lines = []
        for mount in record["mounts"]:
            if inner == '{{printf "%s|%s\\n" .Source .Destination}}':
                lines.append(f"{mount['src']}|{mount['dst']}\n")
            elif inner == "{{println .Source}}":
                lines.append(f"{mount['src']}\n")
            else:
                raise ShimError(f"unsupported inspect template {template!r}")
        return "".join(lines)
    if template == "{{.State.Running}}":
        return "true\n"
    if template == "{{.HostConfig.NetworkMode}}":
        return f"{record.get('network', 'bridge')}\n"
    raise ShimError(f"unsupported inspect template {template!r}")


def cmd_inspect(args: list[str]) -> int:
    values, _, names = parse_options(args, set(), {"-f", "--format", "--type"})
    template = (values.get("-f") or values.get("--format") or [None])[-1]
    if not names:
        raise ShimError("docker inspect requires a name")
    output = []
    for name in names:
        record = load_container(name)
        if record is None:
            if normalize_image(name) in known_images() and template is None:
                output.append(json.dumps([{"RepoTags": [name], "Comment": "Modal image"}]) + "\n")
                continue
            return _fail(f"Error: No such object: {name}", 1)
        if template is None:
            output.append(json.dumps([{"Name": f"/{name}", "Config": {"Image": record["image"]}}]) + "\n")
        else:
            output.append(render_inspect(record, template))
    sys.stdout.write("".join(output))
    return 0


def cmd_image(args: list[str]) -> int:
    if not args or args[0] != "inspect":
        raise ShimError(f"docker image {' '.join(args)} is not supported")
    missing = [name for name in args[1:] if not name.startswith("-") and normalize_image(name) not in known_images()]
    if missing:
        return _fail(f"Error: No such image: {missing[0]}", 1)
    print(json.dumps([{"RepoTags": args[1:], "Comment": "built by Modal from tools/modal_v2d/images.py"}]))
    return 0


def cmd_ps(args: list[str]) -> int:
    values, flags, rest = parse_options(args, {"-a", "--all", "-q", "--quiet"}, {"--format", "--filter", "-f"})
    if rest:
        raise ShimError(f"unexpected docker ps arguments {rest}")
    template = (values.get("--format") or ["{{.Names}}"])[-1]
    if template != "{{.Names}}":
        raise ShimError(f"unsupported docker ps format {template!r}")
    names = [record["name"] for record in all_containers()]
    for expression in values.get("--filter", []) + values.get("-f", []):
        key, _, pattern = expression.partition("=")
        if key != "name":
            raise ShimError(f"unsupported docker ps filter {expression!r}")
        regex = re.compile(pattern.replace("^/", "^"))
        names = [name for name in names if regex.search(name)]
    for name in names:
        print(name)
    return 0


def cmd_stop_rm(args: list[str], verb: str) -> int:
    _, _, names = parse_options(args, {"-f", "--force"}, {"-t", "--time"})
    status = 0
    for name in names:
        if remove_container(name):
            print(name)
        else:
            status = _fail(f"Error: No such container: {name}", 1)
    return status


def cmd_pull(args: list[str]) -> int:
    """`docker pull <image>` = warm a container of that image ahead of its first call.

    Images are built by Modal, so there is nothing to pull; instead the parent starts a
    no-op call so a container is already up (image loaded) when the real call arrives.
    Fire-and-forget: returns at once, never fails the caller."""
    images = [normalize_image(a) for a in args if not a.startswith("-")]
    for image in images:
        if image not in known_images() or image == local_image():
            continue
        path = os.environ.get("V2D_MODAL_RPC", C.RPC_SOCKET)
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(path)
                request = {"op": "warm", "image": image, "caller": os.environ.get("V2D_MODAL_CALLER")}
                client.sendall((json.dumps(request) + "\n").encode())
                client.makefile("r").readline()
        except OSError as exc:
            print(f"[v2d Modal shim] warm {image} skipped: {exc}", file=sys.stderr)
    return 0


def cmd_build(args: list[str]) -> int:
    values, _, _ = parse_options(
        args,
        {"--load", "--push", "--no-cache", "-q", "--quiet", "--pull"},
        {"-t", "--tag", "-f", "--file", "--platform", "--build-arg", "--target", "--progress"},
    )
    tags = values.get("-t", []) + values.get("--tag", [])
    unknown = [tag for tag in tags if normalize_image(tag) not in known_images()]
    if unknown:
        return _fail(f"no Modal image is defined for {unknown}; add it to tools/modal_v2d/images.py", 1)
    print(f"[v2d Modal shim] skipping docker build of {tags or 'untagged image'}: images are built by Modal")
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        return _fail("usage: docker <command> ...")
    command, args = argv[0], argv[1:]
    try:
        if command in {"version", "info"}:
            print(f"v2d Modal docker shim (local image: {local_image() or 'none'})")
            return 0
        if command == "run":
            return cmd_run(args)
        if command == "exec":
            return cmd_exec(args)
        if command == "inspect":
            return cmd_inspect(args)
        if command == "container" and args[:1] == ["inspect"]:
            return cmd_inspect(args[1:])
        if command == "image":
            return cmd_image(args)
        if command == "ps":
            return cmd_ps(args)
        if command in {"stop", "rm"}:
            return cmd_stop_rm(args, command)
        if command == "pull":
            return cmd_pull(args)
        if command == "build":
            return cmd_build(args)
        if command == "buildx" and args[:1] == ["build"]:
            return cmd_build(args[1:])
        return _fail(f"`docker {command}` is not supported on Modal")
    except ShimError as exc:
        return _fail(str(exc))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
