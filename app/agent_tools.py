# pylint: disable=missing-module-docstring,missing-function-docstring,too-many-return-statements
"""Локальный оператор tools для агентного цикла (по мотивам notioncode_mcp runtime).

Инструменты: list_files / read_file / write_file / edit_file / run_shell.
Все пути ограничены корнем TOOLS_ROOT (по умолчанию ~), как в референсе.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

MAX_READ_BYTES = 2_000_000
DEFAULT_READ_BYTES = 500_000
MAX_LIST_ENTRIES = 2000


def tool_root() -> Path:
    """Корень, в пределах которого разрешены все файловые операции."""
    return Path(os.getenv("TOOLS_ROOT", os.path.expanduser("~"))).resolve()


def tools_enabled() -> bool:
    return os.getenv("TOOLS_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")


def resolve_path(input_path: Any) -> Path:
    """Резолв пути с жёсткой проверкой, что он внутри TOOLS_ROOT."""
    root = tool_root()
    candidate = Path(str(input_path or "."))
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if candidate != root and not str(candidate).startswith(f"{root}{os.sep}"):
        raise ValueError(f"Path is outside TOOLS_ROOT ({root}): {input_path}")
    return candidate


def _list_files(args: dict[str, Any]) -> str:
    directory = resolve_path(args.get("directory") or ".")
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    root = tool_root()
    entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    lines: list[str] = []
    for entry in entries[:MAX_LIST_ENTRIES]:
        try:
            rel = str(entry.relative_to(root))
        except ValueError:  # pragma: no cover - root сам себя
            rel = str(entry)
        lines.append(("[dir]  " if entry.is_dir() else "       ") + rel)
    if len(entries) > MAX_LIST_ENTRIES:
        lines.append(f"... and {len(entries) - MAX_LIST_ENTRIES} more entries")
    return "\n".join(lines) or "(empty)"


def _read_file(args: dict[str, Any]) -> str:
    max_bytes = int(args.get("max_bytes") or DEFAULT_READ_BYTES)
    max_bytes = max(1, min(max_bytes, MAX_READ_BYTES))
    file_path = resolve_path(args.get("file_path"))
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.is_dir():
        raise IsADirectoryError(f"Path is a directory: {file_path}")
    data = file_path.read_bytes()
    if len(data) > max_bytes:
        raise ValueError(
            f"File exceeds max_bytes ({len(data)} > {max_bytes}): {args.get('file_path')}"
        )
    return data.decode("utf-8", errors="replace")


def _write_file(args: dict[str, Any]) -> str:
    file_path = resolve_path(args.get("file_path"))
    content = str(args.get("content") or "")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    rel = file_path.relative_to(tool_root())
    return f"Wrote {rel} ({len(content.encode('utf-8'))} bytes)."


def _edit_file(args: dict[str, Any]) -> str:
    file_path = resolve_path(args.get("file_path"))
    old_text = str(args.get("old_text") or "")
    new_text = str(args.get("new_text") or "")
    replace_all = bool(args.get("replace_all") or False)
    if not old_text:
        raise ValueError("old_text is required")
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    current = file_path.read_text(encoding="utf-8", errors="replace")
    count = current.split(old_text).count("") - 1 if False else current.count(old_text)
    if count == 0:
        raise ValueError(f"old_text was not found in {args.get('file_path')}")
    if not replace_all and count != 1:
        raise ValueError(
            f"old_text occurs {count} times; set replace_all=true or provide a larger fragment"
        )
    updated = current.replace(old_text, new_text) if replace_all else current.replace(
        old_text, new_text, 1
    )
    file_path.write_text(updated, encoding="utf-8")
    rel = file_path.relative_to(tool_root())
    return f"Edited {rel} ({count if replace_all else 1} replacement)."


def _run_shell(args: dict[str, Any]) -> str:
    command = str(args.get("command") or "").strip()
    if not command:
        raise ValueError("command is required")
    cwd = resolve_path(args.get("cwd") or ".")
    if not cwd.is_dir():
        raise NotADirectoryError(f"cwd is not a directory: {cwd}")
    timeout_ms = int(args.get("timeout_ms") or 30_000)
    timeout_s = max(1, min(timeout_ms, 120_000)) / 1000.0
    proc = subprocess.run(  # noqa: S602 - инструмент оператора, как в референсе
        command,
        shell=True,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    parts = [f"exit_code: {proc.returncode}"]
    if out:
        parts.append(f"stdout:\n{out}")
    if err:
        parts.append(f"stderr:\n{err}")
    return "\n".join(parts)


TOOL_HANDLERS = {
    "list_files": _list_files,
    "read_file": _read_file,
    "write_file": _write_file,
    "edit_file": _edit_file,
    "run_shell": _run_shell,
}


def execute_tool(name: str, args: dict[str, Any]) -> str:
    """Выполнить tool и вернуть текстовый результат (или пробросить исключение)."""
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        raise ValueError(
            f"Unknown tool '{name}'. Available: {', '.join(sorted(TOOL_HANDLERS))}, final"
        )
    return handler(args or {})
