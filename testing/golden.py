#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
import difflib
import shlex
from pathlib import Path

try:
    import tomllib
except ImportError as exc:
    raise SystemExit("Python 3.11+ required for tomllib") from exc


INFO_FILE = "info.toml"

ERROR_PARSING = -1
IGNORE = 0  # Known bug / known to fail, ignore for now
SHOULD_PASS = 1  # Should pass
SHOULD_FAIL = 2  # Test for failure

@dataclass
class TestConfig:
    # Name for the test
    name: str
    # A description for the test
    description: str | None
    source: str

    # Optional runner override (string or list of args)
    runner: list[str] | None

    # Optional per-stage flag overrides
    flags: dict[str, list[str]]

    # Expected out when parsing
    test_parse: int
    # Expected out for mir generation
    test_mir: int
    # Expected out for vm run
    test_vm: int


def normalize(text: str) -> str:
    return text.replace("\r\n", "\n").rstrip()


def parse_expected_out(text: str) -> int:
    match text.strip().lower():
        case "pass":
            return SHOULD_PASS
        case "fail":
            return SHOULD_FAIL
        case "ignore":
            return IGNORE
        case _:
            raise Exception("Could not parse file")


def parse_bool_like(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "true":
            return True
        if v == "false":
            return False
    return None


def parse_str_list(value) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return shlex.split(value)
    return None


def parse_config(path: Path) -> TestConfig:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    test = data.get("test")
    if not isinstance(test, dict):
        raise ValueError(f"Missing [test] table in {path}")

    name = str(test.get("name", path.parent.name))
    description = test.get("description", None)
    source = str(test.get("source", "main.cm"))

    runner = parse_str_list(test.get("runner"))

    flags: dict[str, list[str]] = {}
    raw_flags = test.get("flags")
    if isinstance(raw_flags, dict):
        for key, val in raw_flags.items():
            parsed = parse_str_list(val)
            if parsed is not None:
                flags[str(key)] = parsed

    test_mir_str = test.get("mir", "pass")
    test_mir = parse_expected_out(test_mir_str)

    test_parse_str = test.get("parser", "pass")
    test_parse = parse_expected_out(test_parse_str)

    test_vm_str = test.get("vm", "ignore")
    test_vm = parse_expected_out(test_vm_str)

    return TestConfig(
        name=name,
        description=description,
        source=source,
        runner=runner,
        flags=flags,
        test_mir=test_mir,
        test_parse=test_parse,
        test_vm=test_vm,
    )


def find_info_files(root: Path) -> list[Path]:
    return sorted(root.rglob(INFO_FILE))


def resolve_path(test_dir: Path, rel: str) -> Path:
    path = (test_dir / rel).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path


def find_binary() -> Path | None:
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "target" / "debug" / "calamars_cli",
        root / "target" / "release" / "calamars_cli",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def ensure_binary() -> Path:
    bin_path = find_binary()
    if bin_path:
        return bin_path
    root = Path(__file__).resolve().parents[1]
    print("==== BUILDING calamars_cli")
    result = subprocess.run(
        ["cargo", "build", "-p", "calamars_cli"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit("Failed to build calamars_cli")
    bin_path = find_binary()
    if not bin_path:
        raise SystemExit("Binary not found after build")
    return bin_path


def run_cmd(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def compare_output(actual: str, expected_path: Path) -> tuple[bool, str]:
    expected = expected_path.read_text(encoding="utf-8")
    a = normalize(actual)
    e = normalize(expected)
    if a == e:
        return True, ""
    diff = "\n".join(
        difflib.unified_diff(
            e.splitlines(),
            a.splitlines(),
            fromfile=f"expected/{expected_path.name}",
            tofile="actual",
            lineterm="",
        )
    )
    return False, f"Expected {expected_path.name} to match output\n{diff}"


def resolve_runner(cfg: TestConfig) -> list[str] | None:
    env_runner = os.environ.get("CALAMARS_RUNNER")
    if env_runner:
        return parse_str_list(env_runner)
    return cfg.runner


def build_cmd(
    base_cmd: list[str],
    flags: list[str],
    source_path: Path,
) -> list[str]:
    cmd = base_cmd + flags
    source_str = str(source_path)
    if any("{source}" in part for part in cmd):
        return [part.replace("{source}", source_str) for part in cmd]
    return cmd + [source_str]


def expected_files(test_dir: Path, stage: str) -> tuple[Path | None, Path | None]:
    out_path = test_dir / f"{stage}.out"
    err_path = test_dir / f"{stage}.err"
    return (out_path if out_path.exists() else None, err_path if err_path.exists() else None)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    testing_dir = root / "testing"
    info_files = find_info_files(testing_dir)
    if not info_files:
        print(f"No {INFO_FILE} files found under {testing_dir}")
        return 1

    bin_path = None
    failures = 0
    total = 0

    for info_path in info_files:
        try:
            cfg = parse_config(info_path)
        except Exception as exc:
            print(f"==== FAIL {info_path} did not parse {exc}")
            continue

        test_dir = info_path.parent
        source_path = resolve_path(test_dir, cfg.source)

        runner = resolve_runner(cfg)
        if runner is None:
            if bin_path is None:
                bin_path = ensure_binary()
            base_cmd = [str(bin_path), "build"]
        else:
            base_cmd = runner

        steps: list[tuple[str, list[str], int]] = []
        if cfg.test_parse != IGNORE:
            flags = cfg.flags.get("parse", [])
            steps.append(("parse", flags, cfg.test_parse))
        if cfg.test_mir != IGNORE:
            flags = cfg.flags.get("mir", ["--emit-mir"])
            steps.append(("mir", flags, cfg.test_mir))
        if cfg.test_vm != IGNORE:
            flags = cfg.flags.get("vm", ["--run-vm"])
            steps.append(("vm", flags, cfg.test_vm))

        for label, flags, expected_out in steps:
            total += 1
            print(f"==== RUNNING {cfg.name} [{label}]")
            cmd = build_cmd(base_cmd, flags, source_path)
            result = run_cmd(cmd, cwd=root)

            if expected_out == SHOULD_PASS and result.returncode != 0:
                failures += 1
                print(f"==== FAIL {cfg.name} [{label}] (non-zero exit)")
                print(result.stdout)
                print(result.stderr, file=sys.stderr)
                continue

            if expected_out == SHOULD_FAIL and result.returncode == 0:
                failures += 1
                print(f"==== FAIL {cfg.name} [{label}] (expected failure)")
                continue

            out_path, err_path = expected_files(test_dir, label)
            if out_path:
                ok, msg = compare_output(result.stdout, out_path)
                if not ok:
                    failures += 1
                    print(f"==== FAIL {cfg.name} [{label}] (stdout mismatch)")
                    print(msg)
                    continue
            if err_path:
                ok, msg = compare_output(result.stderr, err_path)
                if not ok:
                    failures += 1
                    print(f"==== FAIL {cfg.name} [{label}] (stderr mismatch)")
                    print(msg)
                    continue

            print(f"==== PASS {cfg.name} [{label}]")

    print(f"==== DONE {total} checks, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
