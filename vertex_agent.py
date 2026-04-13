import argparse
import difflib
import html
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from google import genai
from google.auth import default as google_auth_default
from google.genai import types
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape as markup_escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

console = Console()
DRY_RUN: bool = False
MODEL_NAME = "gemini-2.5-pro"
LOCATION = "us-central1"
TEMPERATURE = 0.3
_SYSTEM_INSTRUCTION: str | None = None


_DEFAULT_SYSTEM_PROMPT = """\
You are an expert software engineering assistant running as a local agentic REPL.
You have access to tools that can read and write files, execute shell commands, \
search file contents, and browse the web. Use them proactively to complete tasks \
accurately rather than guessing.

## Tool usage guidelines
- Always call glob_files or list_directory to orient yourself before assuming a \
project's structure.
- Always call grep_files to locate relevant code before reading entire files.
- Always call read_file before touching any existing file.
- Prefer edit_file over write_file for changes to existing files — it is faster, \
uses fewer tokens, and is less likely to introduce unintended changes. Only use \
write_file when creating a new file or replacing it entirely.
- Never reconstruct file contents from memory.
- Prefer execute_bash_background for commands that may take more than a few seconds \
(builds, installs, test suites). Poll with get_job_output until they finish.
- Use fetch_url when you need current documentation, package versions, or \
information beyond your training data.

## Behaviour
- Make the smallest correct change. Do not refactor or reformat code outside the \
scope of the request.
- When a task requires multiple steps, state your plan briefly before executing it.
- If something is unclear, ask a single focused question rather than proceeding \
on assumptions.
- Never delete or overwrite files without explicit instruction to do so.
"""

# Tracks files read in the current session to enforce read-before-write.
_read_files: set[str] = set()

# Background job registry: job_id -> {process, command, start_time, paths, file handles}
_background_jobs: dict[str, dict] = {}
_job_counter: int = 0


def _confirm_execution(prompt: str) -> bool:
    response = console.input(f"{prompt} — Allow execution? [bold dodger_blue1]\\[y/n][/bold dodger_blue1]: ").strip().lower()
    return response in ("y", "yes")


def read_file(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        _read_files.add(str(Path(filepath).resolve()))
        return content
    except Exception as e:
        return f"ERROR: {type(e).__name__} - {e}"


def _render_file_diff(
    diff_lines: list[str],
    filepath: str,
    is_new_file: bool,
    added: int,
    removed: int,
) -> None:
    """Render a file diff in Claude Code style.

    • Named header with Create/Update action
    • Tree-indented summary line (added / removed counts)
    • Per-hunk blocks with real file line numbers in the gutter
    • Full-row green background for added lines, red for removed
    • Dim context lines (unchanged)
    • `···` separator between non-adjacent hunks
    • Truncates after MAX_CHANGED changed lines
    """
    action = "Create" if is_new_file else "Update"
    console.print(f"\n[bold white]• {action}([dodger_blue1]{filepath}[/dodger_blue1])[/bold white]")

    parts: list[str] = []
    if added:
        parts.append(f"[green]+{added} line{'s' if added != 1 else ''} added[/green]")
    if removed:
        parts.append(f"[red]-{removed} line{'s' if removed != 1 else ''} removed[/red]")
    summary = ", ".join(parts) if parts else "[gray50]no changes[/gray50]"
    console.print(f"  [gray50]└[/gray50] {summary}")

    if not diff_lines:
        return

    # ── Parse unified diff into hunks ──────────────────────────────────────
    hunks: list[dict] = []
    current_hunk: dict | None = None
    for raw in diff_lines:
        if raw.startswith("---") or raw.startswith("+++"):
            continue
        m = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
        if m:
            current_hunk = {
                "old_start": int(m.group(1)),
                "new_start": int(m.group(2)),
                "lines": [],
            }
            hunks.append(current_hunk)
        elif current_hunk is not None and not raw.startswith("\\"):
            current_hunk["lines"].append(raw)

    if not hunks:
        return

    # ── Gutter width: enough for the largest line number in the diff ───────
    max_lineno = max(
        h["new_start"] + sum(1 for ln in h["lines"] if not ln.startswith("-"))
        for h in hunks
    )
    gutter_w = max(len(str(max_lineno)), 3)

    MAX_CHANGED = 80
    total_changed = 0
    truncated = False

    for hunk_idx, hunk in enumerate(hunks):
        if hunk_idx > 0:
            # Separator between non-adjacent change groups
            console.print(Text(f" {'···':>{gutter_w + 4}}", style="gray50"))

        old_ln = hunk["old_start"]
        new_ln = hunk["new_start"]

        for raw in hunk["lines"]:
            if total_changed >= MAX_CHANGED:
                truncated = True
                break

            t = Text(overflow="fold")
            raw_content = raw[1:].rstrip("\r\n")
            if raw.startswith("+"):
                gn = str(new_ln).rjust(gutter_w)
                t.append(f" {gn} ", style="bold bright_green on color(22)")
                t.append("+ ", style="bright_green on color(22)")
                t.append(raw_content, style="white on color(22)")
                new_ln += 1
                total_changed += 1
            elif raw.startswith("-"):
                gn = str(old_ln).rjust(gutter_w)
                t.append(f" {gn} ", style="bold bright_red on color(52)")
                t.append("- ", style="bright_red on color(52)")
                t.append(raw_content, style="white on color(52)")
                old_ln += 1
                total_changed += 1
            else:
                # context line (starts with a space)
                gn = str(new_ln).rjust(gutter_w)
                t.append(f" {gn}   ", style="#d0d0d0")
                t.append(raw_content, style="#d0d0d0")
                old_ln += 1
                new_ln += 1

            console.print(t)

        if truncated:
            break

    if truncated:
        remaining = (added + removed) - total_changed
        console.print(
            f"  [dim]… {remaining} more change{'s' if remaining != 1 else ''} not shown[/dim]"
        )


def write_file(filepath: str, content: str) -> str:
    if DRY_RUN:
        console.print(f"[dim][DRY RUN] Would have written {len(content)} bytes to {filepath!r}[/dim]")
        return "Success"

    fp = Path(filepath)
    is_new_file = not fp.exists()
    old_content = ""
    if not is_new_file:
        if str(fp.resolve()) not in _read_files:
            return (
                f"ERROR: ReadRequired - '{filepath}' already exists but has not been read "
                "in this session. Call read_file first so you have the current contents."
            )
        try:
            old_content = fp.read_text(encoding="utf-8")
        except Exception:
            pass

    if not _confirm_execution(f"[bold #ffffd7]write_file({filepath!r})[/bold #ffffd7]"):
        return "Execution blocked by user."

    try:
        fp.write_text(content, encoding="utf-8")
    except Exception as e:
        return f"ERROR: {type(e).__name__} - {e}"

    old_lines = old_content.splitlines(keepends=True)
    new_lines = content.splitlines(keepends=True)
    diff_lines = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filepath}", tofile=f"b/{filepath}",
        lineterm="",
        n=3,
    ))
    added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))

    _render_file_diff(diff_lines, filepath, is_new_file, added, removed)

    return f"Successfully wrote {len(content)} bytes to {filepath!r}."


def edit_file(filepath: str, old_string: str, new_string: str) -> str:
    fp = Path(filepath)
    if not fp.exists():
        return f"ERROR: FileNotFound - '{filepath}' does not exist. Use write_file to create new files."
    if str(fp.resolve()) not in _read_files:
        return (
            f"ERROR: ReadRequired - '{filepath}' must be read before it can be edited. "
            "Call read_file first."
        )
    if DRY_RUN:
        console.print(f"[dim][DRY RUN] Would have edited {filepath!r}[/dim]")
        return "Success"

    try:
        content = fp.read_text(encoding="utf-8")
    except Exception as e:
        return f"ERROR: {type(e).__name__} - {e}"

    count = content.count(old_string)
    if count == 0:
        return (
            f"ERROR: StringNotFound - The exact string was not found in '{filepath}'. "
            "Call read_file again to get the latest contents, then retry with the exact text including correct whitespace and indentation."
        )
    if count > 1:
        return (
            f"ERROR: AmbiguousMatch - The string appears {count} times in '{filepath}'. "
            "Include more surrounding context to make the match unique."
        )

    new_content = content.replace(old_string, new_string, 1)

    if not _confirm_execution(f"[bold #ffffd7]edit_file({filepath!r})[/bold #ffffd7]"):
        return "Execution blocked by user."

    try:
        fp.write_text(new_content, encoding="utf-8")
    except Exception as e:
        return f"ERROR: {type(e).__name__} - {e}"

    old_lines = content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff_lines = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filepath}", tofile=f"b/{filepath}",
        lineterm="", n=3,
    ))
    added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
    _render_file_diff(diff_lines, filepath, False, added, removed)

    return f"Successfully edited '{filepath}' ({added} lines added, {removed} removed)."


def list_directory(path: str = ".") -> str:
    _ICON_MAP = {".py": "🐍", ".txt": "📄", ".json": "⚙️", ".md": "📝"}
    _KIND_MAP = {".py": "Python", ".txt": "Text", ".json": "JSON", ".md": "Markdown"}

    try:
        entries = sorted(os.listdir(path))
    except Exception as e:
        return f"ERROR: {type(e).__name__} - {e}"

    if not entries:
        console.print(f"[gray50](empty directory: {path!r})[/gray50]")
        return f"(empty directory: {path!r})"

    tbl = Table(
        border_style="dodger_blue1", show_header=True,
        header_style="bold gray93", box=box.SIMPLE_HEAD,
    )
    tbl.add_column("", no_wrap=True, width=3)
    tbl.add_column("Name", no_wrap=True)
    tbl.add_column("Size", justify="right", style="gray50")
    tbl.add_column("Kind", style="gray50")

    for name in entries:
        full = os.path.join(path, name)
        is_dir = os.path.isdir(full)
        ext = Path(name).suffix.lower()
        icon = "📁" if is_dir else _ICON_MAP.get(ext, "📄")
        kind = "Folder" if is_dir else _KIND_MAP.get(ext, "File")
        if is_dir:
            sz = "—"
        else:
            try:
                b = os.path.getsize(full)
                sz = f"{b / 1024:.1f} KB" if b >= 1024 else f"{b} B"
            except OSError:
                sz = "?"
        name_display = (
            f"[bold medium_purple1]{name}[/bold medium_purple1]"
            if is_dir else f"[white]{name}[/white]"
        )
        tbl.add_row(icon, name_display, sz, kind)

    console.print(tbl)
    return "\n".join(entries)


def execute_bash(command: str) -> str:
    if DRY_RUN:
        console.print(f"[dim][DRY RUN] Would have executed: {command!r}[/dim]")
        return "Success"
    if not _confirm_execution(f"[bold #ffffd7]execute_bash({command!r})[/bold #ffffd7]"):
        return "Execution blocked by user."
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return (
                f"ERROR: NonZeroExit - Command exited with code {result.returncode}.\n"
                f"STDERR: {result.stderr.strip() or '(none)'}\n"
                f"STDOUT: {result.stdout.strip() or '(none)'}"
            )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: TimeoutExpired - Command timed out after 60 seconds."
    except Exception as e:
        return f"ERROR: {type(e).__name__} - {e}"


def execute_bash_background(command: str) -> str:
    global _job_counter
    if DRY_RUN:
        console.print(f"[dim][DRY RUN] Would have started background job: {command!r}[/dim]")
        return "Success"
    if not _confirm_execution(f"[bold #ffffd7]execute_bash_background({command!r})[/bold #ffffd7]"):
        return "Execution blocked by user."

    _job_counter += 1
    job_id = str(_job_counter)
    stdout_path = os.path.join(tempfile.gettempdir(), f"vagent_job_{job_id}.out")
    stderr_path = os.path.join(tempfile.gettempdir(), f"vagent_job_{job_id}.err")

    try:
        stdout_f = open(stdout_path, "w", encoding="utf-8")
        stderr_f = open(stderr_path, "w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=stdout_f,
            stderr=stderr_f,
            text=True,
        )
        _background_jobs[job_id] = {
            "process": process,
            "command": command,
            "start_time": time.monotonic(),
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "stdout_f": stdout_f,
            "stderr_f": stderr_f,
        }
        return f"Job {job_id} started (PID {process.pid}). Use get_job_output('{job_id}') to check status."
    except Exception as e:
        try:
            stdout_f.close()
            stderr_f.close()
        except Exception:
            pass
        return f"ERROR: {type(e).__name__} - {e}"


def get_job_output(job_id: str) -> str:
    job = _background_jobs.get(job_id)
    if job is None:
        available = list(_background_jobs.keys())
        return f"ERROR: No job with ID '{job_id}'. Active jobs: {available or 'none'}"

    process = job["process"]
    returncode = process.poll()
    elapsed = time.monotonic() - job["start_time"]

    if returncode is not None:
        try:
            job["stdout_f"].close()
            job["stderr_f"].close()
        except Exception:
            pass

    try:
        stdout = Path(job["stdout_path"]).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        stdout = "(could not read stdout)"
    try:
        stderr = Path(job["stderr_path"]).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        stderr = ""

    if returncode is None:
        status = f"RUNNING ({elapsed:.1f}s elapsed)"
    else:
        status = f"DONE (exit code {returncode}, {elapsed:.1f}s)"
        try:
            os.unlink(job["stdout_path"])
            os.unlink(job["stderr_path"])
        except Exception:
            pass
        del _background_jobs[job_id]

    lines = [f"Job {job_id}: {job['command']!r}", f"Status: {status}"]
    if stdout.strip():
        lines.append(f"STDOUT:\n{stdout.rstrip()}")
    if stderr.strip():
        lines.append(f"STDERR:\n{stderr.rstrip()}")
    if not stdout.strip() and not stderr.strip():
        lines.append("(no output yet)")
    return "\n".join(lines)


def git_status() -> str:
    try:
        r = subprocess.run(["git", "status", "--short", "--branch"], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return f"ERROR: {r.stderr.strip() or 'git status failed'}"
        return r.stdout.strip() or "(clean working tree)"
    except Exception as e:
        return f"ERROR: {type(e).__name__} - {e}"


def git_diff(filepath: str = "") -> str:
    try:
        cmd = ["git", "diff"]
        if filepath:
            cmd.append(filepath)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return f"ERROR: {r.stderr.strip()}"
        output = r.stdout.strip()
        MAX = 8_000
        if len(output) > MAX:
            output = output[:MAX] + f"\n... (truncated at {MAX} chars)"
        return output or "(no unstaged changes)"
    except Exception as e:
        return f"ERROR: {type(e).__name__} - {e}"


def git_log(max_entries: int = 10) -> str:
    try:
        r = subprocess.run(
            ["git", "log", f"--max-count={max_entries}", "--oneline", "--decorate"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return f"ERROR: {r.stderr.strip()}"
        return r.stdout.strip() or "(no commits)"
    except Exception as e:
        return f"ERROR: {type(e).__name__} - {e}"


def git_commit(message: str, files: list = None) -> str:
    if DRY_RUN:
        console.print(f"[dim][DRY RUN] Would have committed with message {message!r}[/dim]")
        return "Success"
    if not _confirm_execution(f"[bold #ffffd7]git_commit({message!r})[/bold #ffffd7]"):
        return "Execution blocked by user."
    try:
        stage_cmd = ["git", "add"] + (files if files else ["-A"])
        r = subprocess.run(stage_cmd, capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return f"ERROR: git add failed: {r.stderr.strip()}"
        r = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return f"ERROR: git commit failed: {r.stderr.strip()}"
        return r.stdout.strip()
    except Exception as e:
        return f"ERROR: {type(e).__name__} - {e}"


def glob_files(pattern: str, path: str = ".") -> str:
    try:
        base = Path(path)
        matches = sorted(str(p) for p in base.glob(pattern))
        if not matches:
            return f"No files matched pattern {pattern!r} in {path!r}"
        MAX = 500
        truncated = len(matches) > MAX
        result = "\n".join(matches[:MAX])
        if truncated:
            result += f"\n... ({len(matches) - MAX} more results not shown)"
        return result
    except Exception as e:
        return f"ERROR: {type(e).__name__} - {e}"


def grep_files(pattern: str, path: str = ".", glob: str = "**/*") -> str:
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"ERROR: Invalid regex - {e}"
    try:
        base = Path(path)
        results: list[str] = []
        MAX = 200
        for filepath in sorted(base.glob(glob)):
            if not filepath.is_file():
                continue
            try:
                lines = filepath.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue
            for i, line in enumerate(lines, 1):
                if regex.search(line):
                    results.append(f"{filepath}:{i}: {line}")
                    if len(results) >= MAX:
                        break
            if len(results) >= MAX:
                break
        if not results:
            return f"No matches for pattern {pattern!r}"
        if len(results) == MAX:
            results.append(f"... (result limit of {MAX} reached, try a more specific pattern or path)")
        return "\n".join(results)
    except Exception as e:
        return f"ERROR: {type(e).__name__} - {e}"


class _TextExtractor(HTMLParser):
    """Strip HTML tags and decode entities, keeping plain text."""
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "head"}:
            self._skip = True

    def handle_endtag(self, tag):
        if tag in {"script", "style", "head"}:
            self._skip = False
        if tag in {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse excessive whitespace while preserving paragraph breaks
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def fetch_url(url: str) -> str:
    MAX_CHARS = 12_000
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; vagent/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(MAX_CHARS * 4)  # read extra to allow for HTML overhead

        text = raw.decode("utf-8", errors="replace")

        if "text/html" in content_type:
            parser = _TextExtractor()
            parser.feed(html.unescape(text))
            text = parser.get_text()

        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + f"\n\n... (truncated at {MAX_CHARS} chars)"
        return text
    except Exception as e:
        return f"ERROR: {type(e).__name__} - {e}"


# ---------------------------------------------------------------------------
# Vertex AI Tool definitions
# ---------------------------------------------------------------------------

_read_file_declaration = types.FunctionDeclaration(
    name="read_file",
    description="Read the full text contents of a file on the local filesystem.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "filepath": types.Schema(
                type=types.Type.STRING,
                description="Absolute or relative path to the file to read.",
            ),
        },
        required=["filepath"],
    ),
)

_write_file_declaration = types.FunctionDeclaration(
    name="write_file",
    description=(
        "Write (overwrite) a file on the local filesystem with the given content. "
        "The user will be prompted to confirm before the write occurs."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "filepath": types.Schema(
                type=types.Type.STRING,
                description="Absolute or relative path to the file to write.",
            ),
            "content": types.Schema(
                type=types.Type.STRING,
                description="Full text content to write into the file.",
            ),
        },
        required=["filepath", "content"],
    ),
)

_list_directory_declaration = types.FunctionDeclaration(
    name="list_directory",
    description="List the files and folders inside a directory on the local filesystem.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "path": types.Schema(
                type=types.Type.STRING,
                description="Absolute or relative path to the directory. Defaults to '.' (current directory).",
            ),
        },
    ),
)

_execute_bash_declaration = types.FunctionDeclaration(
    name="execute_bash",
    description=(
        "Execute a shell command on the local machine and return its stdout/stderr. "
        "The user will be prompted to confirm before the command runs."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "command": types.Schema(
                type=types.Type.STRING,
                description="The shell command to execute.",
            ),
        },
        required=["command"],
    ),
)

_execute_bash_background_declaration = types.FunctionDeclaration(
    name="execute_bash_background",
    description=(
        "Start a shell command in the background without blocking. Returns a job ID immediately. "
        "Use get_job_output to check status and retrieve output. "
        "The user will be prompted to confirm before the command runs."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "command": types.Schema(
                type=types.Type.STRING,
                description="The shell command to run in the background.",
            ),
        },
        required=["command"],
    ),
)

_get_job_output_declaration = types.FunctionDeclaration(
    name="get_job_output",
    description=(
        "Check the status and output of a background job started with execute_bash_background. "
        "Returns status (RUNNING or DONE), stdout, and stderr."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "job_id": types.Schema(
                type=types.Type.STRING,
                description="The job ID returned by execute_bash_background.",
            ),
        },
        required=["job_id"],
    ),
)

_edit_file_declaration = types.FunctionDeclaration(
    name="edit_file",
    description=(
        "Make a targeted edit to an existing file by replacing an exact string with new text. "
        "Prefer this over write_file for modifying existing files — it is faster and token-efficient. "
        "The old_string must match the file contents exactly (including whitespace and indentation). "
        "The file must have been read with read_file in this session first."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "filepath": types.Schema(type=types.Type.STRING, description="Path to the file to edit."),
            "old_string": types.Schema(type=types.Type.STRING, description="The exact string to find and replace. Must be unique within the file."),
            "new_string": types.Schema(type=types.Type.STRING, description="The string to replace it with."),
        },
        required=["filepath", "old_string", "new_string"],
    ),
)

_git_status_declaration = types.FunctionDeclaration(
    name="git_status",
    description="Show the working tree status (staged, unstaged, and untracked files).",
    parameters=types.Schema(type=types.Type.OBJECT, properties={}),
)

_git_diff_declaration = types.FunctionDeclaration(
    name="git_diff",
    description="Show unstaged changes as a unified diff. Optionally scoped to a single file.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "filepath": types.Schema(type=types.Type.STRING, description="Limit diff to this file. Omit for all changes."),
        },
    ),
)

_git_log_declaration = types.FunctionDeclaration(
    name="git_log",
    description="Show recent commit history.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "max_entries": types.Schema(type=types.Type.INTEGER, description="Number of commits to show. Defaults to 10."),
        },
    ),
)

_git_commit_declaration = types.FunctionDeclaration(
    name="git_commit",
    description=(
        "Stage files and create a git commit. "
        "If files is omitted, stages all changes (git add -A). "
        "The user will be prompted to confirm before the commit is made."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "message": types.Schema(type=types.Type.STRING, description="The commit message."),
            "files": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
                description="Specific files to stage. Omit to stage all changes.",
            ),
        },
        required=["message"],
    ),
)

_fetch_url_declaration = types.FunctionDeclaration(
    name="fetch_url",
    description=(
        "Fetch the content of a URL and return it as plain text. "
        "Use this to read documentation, browse search results, or retrieve any web page."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "url": types.Schema(
                type=types.Type.STRING,
                description="The fully-qualified URL to fetch (must start with http:// or https://).",
            ),
        },
        required=["url"],
    ),
)

_glob_files_declaration = types.FunctionDeclaration(
    name="glob_files",
    description=(
        "Find files on the local filesystem matching a glob pattern. "
        "Supports ** for recursive matching (e.g. '**/*.py'). Returns a list of matching paths."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "pattern": types.Schema(
                type=types.Type.STRING,
                description="Glob pattern to match, e.g. '**/*.py' or 'src/*.ts'.",
            ),
            "path": types.Schema(
                type=types.Type.STRING,
                description="Base directory to search from. Defaults to '.' (current directory).",
            ),
        },
        required=["pattern"],
    ),
)

_grep_files_declaration = types.FunctionDeclaration(
    name="grep_files",
    description=(
        "Search file contents for lines matching a regex pattern. "
        "Returns matching lines with their file path and line number."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "pattern": types.Schema(
                type=types.Type.STRING,
                description="Regular expression pattern to search for.",
            ),
            "path": types.Schema(
                type=types.Type.STRING,
                description="Directory or file to search in. Defaults to '.' (current directory).",
            ),
            "glob": types.Schema(
                type=types.Type.STRING,
                description="Glob pattern to filter which files are searched, e.g. '**/*.py'. Defaults to '**/*' (all files).",
            ),
        },
        required=["pattern"],
    ),
)

local_tools = types.Tool(
    function_declarations=[
        _read_file_declaration,
        _write_file_declaration,
        _edit_file_declaration,
        _execute_bash_declaration,
        _execute_bash_background_declaration,
        _get_job_output_declaration,
        _list_directory_declaration,
        _glob_files_declaration,
        _grep_files_declaration,
        _fetch_url_declaration,
        _git_status_declaration,
        _git_diff_declaration,
        _git_log_declaration,
        _git_commit_declaration,
    ]
)

TOOL_DISPATCH = {
    "read_file": lambda args: read_file(**args),
    "write_file": lambda args: write_file(**args),
    "edit_file": lambda args: edit_file(**args),
    "execute_bash": lambda args: execute_bash(**args),
    "execute_bash_background": lambda args: execute_bash_background(**args),
    "get_job_output": lambda args: get_job_output(**args),
    "list_directory": lambda args: list_directory(**args),
    "glob_files": lambda args: glob_files(**args),
    "grep_files": lambda args: grep_files(**args),
    "fetch_url": lambda args: fetch_url(**args),
    "git_status": lambda args: git_status(**args),
    "git_diff": lambda args: git_diff(**args),
    "git_log": lambda args: git_log(**args),
    "git_commit": lambda args: git_commit(**args),
}

# ---------------------------------------------------------------------------
# History compaction
# ---------------------------------------------------------------------------

COMPACT_TOKEN_THRESHOLD = 20_000
_SUMMARIZE_PROMPT = (
    "Summarize the key progress, decisions, and current project state into a single "
    "dense paragraph. Retain all technical details and file paths discussed."
)


def compact_history(
    chat_history: list[types.Content], client: genai.Client
) -> list[types.Content]:
    if len(chat_history) < 2:
        console.print("[dim]Nothing substantial to compact.[/dim]")
        return chat_history

    base = chat_history[:-1] if chat_history[-1].role == "user" else chat_history
    if len(base) < 2:
        console.print("[dim]Nothing substantial to compact.[/dim]")
        return chat_history

    summarize_request = base + [
        types.Content(role="user", parts=[types.Part(text=_SUMMARIZE_PROMPT)])
    ]

    with console.status("[bold dodger_blue1]Compacting history...[/bold dodger_blue1]", spinner="dots"):
        summary_response = client.models.generate_content(
            model=MODEL_NAME,
            contents=summarize_request,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="NONE"),
                ),
            ),
        )

    if (
        not summary_response.candidates
        or not summary_response.candidates[0].content
        or not summary_response.candidates[0].content.parts
    ):
        console.print("[bold red]Compaction failed: model returned an empty response. History unchanged.[/bold red]")
        return chat_history

    summary_text = "\n".join(
        p.text for p in summary_response.candidates[0].content.parts if p.text
    )
    if not summary_text:
        console.print("[bold red]Compaction failed: model returned no text. History unchanged.[/bold red]")
        return chat_history

    last_turn: list[types.Content] = []
    if len(chat_history) >= 2:
        prev, last = chat_history[-2], chat_history[-1]
        prev_is_user_text = prev.role == "user" and any(p.text for p in prev.parts)
        last_is_model_text = last.role == "model" and any(p.text for p in last.parts)
        if prev_is_user_text and last_is_model_text:
            last_turn = [prev, last]

    compacted = [
        types.Content(role="user", parts=[types.Part(text=f"[Summary of prior conversation]:\n{summary_text}")]),
        types.Content(role="model", parts=[types.Part(text="Understood. I have the full context from our previous work.")]),
        *last_turn,
    ]

    console.print(
        f"[gray50]Compacted {len(chat_history)} → {len(compacted)} turns.[/gray50]"
    )
    return compacted


# ---------------------------------------------------------------------------
# Response rendering
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


def render_response(text: str) -> None:
    """Print text, rendering fenced code blocks with syntax highlighting."""
    last_end = 0
    for match in _CODE_BLOCK_RE.finditer(text):
        preceding = text[last_end : match.start()]
        if preceding.strip():
            console.print(Markdown(preceding))
        language = match.group(1) or "text"
        code = match.group(2).rstrip()
        console.print(Syntax(code, language, theme="monokai", line_numbers=True, padding=(1, 2)))
        last_end = match.end()

    trailing = text[last_end:]
    if trailing.strip():
        console.print(Markdown(trailing))


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_vertex() -> tuple[genai.Client, str, str]:
    """Authenticate via ADC, load optional .vagent context, return (client, project_id, vagent_content)."""
    global _SYSTEM_INSTRUCTION

    credentials, project_id = google_auth_default()
    if not project_id:
        project_id = getattr(credentials, "quota_project_id", None)
    if not project_id:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise RuntimeError(
            "Could not auto-detect a GCP project. Set one with:\n"
            "  gcloud auth application-default login --project YOUR_PROJECT\n"
            "or: set GOOGLE_CLOUD_PROJECT=YOUR_PROJECT"
        )

    # Start with the default system prompt, then append any .vagent project context.
    _SYSTEM_INSTRUCTION = _DEFAULT_SYSTEM_PROMPT
    vagent_content = ""
    vagent_path = Path(".vagent")
    if vagent_path.exists():
        try:
            vagent_content = vagent_path.read_text(encoding="utf-8").strip()
            if vagent_content:
                _SYSTEM_INSTRUCTION = (
                    f"{_DEFAULT_SYSTEM_PROMPT}\n\n"
                    f"## Project-specific context\n{vagent_content}"
                )
        except Exception as e:
            console.print(f"[dim yellow]Warning: could not read .vagent: {e}[/dim yellow]")

    client = genai.Client(vertexai=True, project=project_id, location=LOCATION)
    return client, project_id, vagent_content


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_environment_header(project_id: str, vagent_content: str) -> None:
    tool_count = len(local_tools.function_declarations)
    mode = "[bold yellow]DRY RUN[/bold yellow]" if DRY_RUN else "[bold dodger_blue1]Normal[/bold dodger_blue1]"
    vagent_status = "[bold dodger_blue1]Loaded[/bold dodger_blue1]" if vagent_content else "[gray50]None[/gray50]"

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column()
    grid.add_row("Current Path:", os.getcwd())
    grid.add_row("GCP Project:", f"[bold]{project_id}[/bold]")
    grid.add_row("Model:", MODEL_NAME)
    grid.add_row("Temperature:", str(TEMPERATURE))
    grid.add_row("Tool Count:", str(tool_count))
    grid.add_row("Mode:", mode)
    grid.add_row(".vagent:", vagent_status)

    console.print(Panel(grid, title="[bold medium_purple1]Vertex Agent[/bold medium_purple1]", border_style="dodger_blue1", box=box.DOUBLE_EDGE))
    console.print("[gray50]Type [bold medium_purple1]?[/bold medium_purple1] or [bold medium_purple1]/help[/bold medium_purple1] for available commands.\n[/gray50]")


def print_help() -> None:
    cmd_table = Table(border_style="dim", show_header=True, header_style="bold")
    cmd_table.add_column("Command", style="bold dodger_blue1", no_wrap=True)
    cmd_table.add_column("Description")
    cmd_table.add_row("[medium_purple1]/exit[/medium_purple1]", "Quit the agent")
    cmd_table.add_row("[medium_purple1]/clear[/medium_purple1]", "Wipe the conversation history")
    cmd_table.add_row("[medium_purple1]/compact[/medium_purple1]", "Summarise and compress history into a single context turn")
    cmd_table.add_row("[medium_purple1]/help[/medium_purple1]  or  [medium_purple1]?[/medium_purple1]", "Show this help")

    tools_table = Table(border_style="dim", show_header=True, header_style="bold")
    tools_table.add_column("Tool", style="bold #0087ff", no_wrap=True)
    tools_table.add_column("Safety Gate")
    tools_table.add_column("Description")
    for decl in local_tools.function_declarations:
        if decl.name in {"read_file", "list_directory"}:
            gate = "[dim]None (read-only)[/dim]"
        else:
            gate = "[yellow]Confirm [Y/n][/yellow]"
        tools_table.add_row(decl.name, gate, (decl.description or "").split(".")[0])

    console.print(Panel(cmd_table, title="[bold dodger_blue1]Slash Commands[/bold dodger_blue1]", border_style="dim dodger_blue1"))
    console.print(Panel(tools_table, title="[bold medium_purple1]Active Tools[/bold medium_purple1]", border_style="dim medium_purple1"))


# ---------------------------------------------------------------------------
# Agentic REPL
# ---------------------------------------------------------------------------

def _build_prompt_session() -> PromptSession:
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        event.current_buffer.validate_and_handle()

    return PromptSession(
        key_bindings=kb,
        multiline=True,
        prompt_continuation=lambda width, line_number, is_soft_wrap: "... ",
    )


def _generate_with_cancel(
    client: genai.Client,
    contents: list,
    config: types.GenerateContentConfig,
) -> tuple:
    """Run generate_content in a background thread.

    Returns (response, cancelled). Pressing Escape while waiting sets cancelled=True
    and returns (None, True); the background thread finishes silently and is discarded.
    """
    result: list = [None]
    exc: list = [None]
    done = threading.Event()

    def _worker():
        try:
            result[0] = client.models.generate_content(
                model=MODEL_NAME, contents=contents, config=config
            )
        except Exception as e:
            exc[0] = e
        finally:
            done.set()

    threading.Thread(target=_worker, daemon=True).start()

    try:
        import msvcrt  # Windows
        def _escape_pressed() -> bool:
            return msvcrt.kbhit() and msvcrt.getch() == b"\x1b"
    except ImportError:
        import select, tty, termios  # Unix
        def _escape_pressed() -> bool:
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                return bool(select.select([sys.stdin], [], [], 0)[0]) and sys.stdin.read(1) == "\x1b"
            except Exception:
                return False
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

    while not done.wait(timeout=0.05):
        if _escape_pressed():
            return None, True

    if exc[0]:
        raise exc[0]
    return result[0], False


def run_agent(client: genai.Client, project_id: str, vagent_content: str) -> None:
    chat_history: list[types.Content] = []
    print_environment_header(project_id, vagent_content)
    prompt_session = _build_prompt_session()

    try:
        while True:
            # ── User input ────────────────────────────────────────────────
            try:
                console.print()
                console.print(Rule(style="dim medium_purple1"))
                user_text = prompt_session.prompt(
                    HTML("<style fg='#0087ff'><b>❯ </b></style>"),
                ).strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye.[/]")
                break

            if not user_text:
                continue

            # ── Slash commands ────────────────────────────────────────────
            if user_text in {"?", "/help"} or user_text.startswith("/"):
                if user_text in {"?", "/help"}:
                    print_help()
                elif user_text == "/exit":
                    console.print("[dim]Goodbye.[/]")
                    sys.exit(0)
                elif user_text == "/clear":
                    if not chat_history:
                        console.print("[dim]History is already empty.[/dim]")
                    else:
                        chat_history.clear()
                        _read_files.clear()
                        console.print("[bold red]History cleared.[/bold red]")
                elif user_text == "/compact":
                    chat_history = compact_history(chat_history, client)
                else:
                    console.print(
                        f"[gray50]Unknown command: {user_text!r}. "
                        "Available: [medium_purple1]/exit[/medium_purple1], [medium_purple1]/clear[/medium_purple1], [medium_purple1]/compact[/medium_purple1][/gray50]"
                    )
                continue

            chat_history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))

            # ── Agentic loop (handles multi-step tool chains) ─────────────
            turn_start = time.monotonic()
            error_counts: dict[str, int] = {}
            while True:
                config = types.GenerateContentConfig(
                    system_instruction=_SYSTEM_INSTRUCTION,
                    tools=[local_tools],
                    temperature=TEMPERATURE,
                )
                status = console.status(
                    "[bold medium_purple1]Agent is thinking...[/bold medium_purple1]  "
                    "[dim](Esc to cancel)[/dim]",
                    spinner="dots",
                )
                status.start()
                try:
                    response, cancelled = _generate_with_cancel(client, chat_history, config)
                except Exception as api_err:
                    status.stop()
                    chat_history.pop()
                    err_str = str(api_err)
                    status_line = err_str.split("{")[0].strip().rstrip(".")
                    human_msg = getattr(api_err, "message", "") or ""
                    body = f"[bold]{markup_escape(status_line)}[/bold]"
                    if human_msg:
                        body += f"\n\n{markup_escape(human_msg)}"
                    console.print(
                        Panel(body, title="[bold red]✘ API Error[/bold red]", border_style="red")
                    )
                    break
                finally:
                    status.stop()

                if cancelled:
                    chat_history.pop()
                    console.print("[dim]Cancelled.[/dim]")
                    break

                candidate = response.candidates[0]

                # Guard: safety-blocked or otherwise empty responses have no parts.
                if not candidate.content or not candidate.content.parts:
                    console.print(
                        Panel(
                            "The model returned an empty or blocked response.",
                            title="[bold red]✘ Empty Response[/bold red]",
                            border_style="red",
                        )
                    )
                    break

                chat_history.append(candidate.content)

                # In google-genai, part.function_call is None when not a tool call.
                function_calls = [
                    part.function_call
                    for part in candidate.content.parts
                    if part.function_call and part.function_call.name
                ]

                if not function_calls:
                    # ── Text response ─────────────────────────────────────
                    text = "\n".join(
                        p.text for p in candidate.content.parts if p.text
                    )
                    if not text:
                        console.print("[dim]Model returned no text.[/dim]")
                        break
                    console.print()
                    console.print(Rule("[italic gray50]Gemini[/italic gray50]", style="dim medium_purple1"))
                    console.print()
                    render_response(text)
                    console.print()
                    # ── Grounding sources ─────────────────────────────────
                    grounding = getattr(candidate, "grounding_metadata", None)
                    if grounding:
                        chunks = getattr(grounding, "grounding_chunks", None) or []
                        if chunks:
                            console.print("[gray50]Sources:[/gray50]")
                            for chunk in chunks:
                                web = getattr(chunk, "web", None)
                                if web:
                                    title = getattr(web, "title", None) or getattr(web, "uri", "")
                                    uri = getattr(web, "uri", "")
                                    console.print(f"  [gray50]• [link={uri}]{markup_escape(title)}[/link][/gray50]")
                            console.print()
                    total_tokens = response.usage_metadata.total_token_count
                    turn_time = time.monotonic() - turn_start
                    console.print(f"[gray50]✦ Calculated... ({turn_time:.1f}s • {total_tokens:,} tokens)[/gray50]")

                    if total_tokens > COMPACT_TOKEN_THRESHOLD:
                        console.print(
                            f"[dim yellow]Token threshold exceeded "
                            f"({total_tokens:,} > {COMPACT_TOKEN_THRESHOLD:,}). "
                            "Auto-compacting...[/dim yellow]"
                        )
                        chat_history = compact_history(chat_history, client)
                    break

                # ── Tool execution turn ───────────────────────────────────
                _SILENT_TOOLS = {
                    "read_file", "list_directory", "glob_files", "grep_files",
                    "get_job_output", "fetch_url",
                    "git_status", "git_diff", "git_log",
                }
                function_response_parts: list[types.Part] = []
                stuck = False
                for fc in function_calls:
                    fn_name = fc.name
                    fn_args = dict(fc.args)

                    if fn_name == "write_file":
                        fp_display = fn_args.get("filepath", "?")
                        console.print(f"\n[bold white on dodger_blue1] ✎ WRITE [/bold white on dodger_blue1] [bold white] {fp_display} [/bold white]")
                    elif fn_name == "edit_file":
                        fp_display = fn_args.get("filepath", "?")
                        console.print(f"\n[bold white on dodger_blue1] ✎ EDIT [/bold white on dodger_blue1] [bold white] {fp_display} [/bold white]")
                    elif fn_name == "git_commit":
                        msg_display = fn_args.get("message", "?")
                        console.print(f"\n[bold white on green4] ⎇ COMMIT [/bold white on green4] [bold white] {msg_display} [/bold white]")
                    elif fn_name == "execute_bash":
                        cmd_display = fn_args.get("command", "?")
                        console.print(f"\n[bold white on medium_purple1] ❯ RUN [/bold white on medium_purple1] [bold white] {cmd_display} [/bold white]")
                    elif fn_name == "execute_bash_background":
                        cmd_display = fn_args.get("command", "?")
                        console.print(f"\n[bold white on gray30] ❯ BG [/bold white on gray30] [bold white] {cmd_display} [/bold white]")
                    elif fn_name in {"read_file", "glob_files", "grep_files", "get_job_output",
                                     "git_status", "git_diff", "git_log", "fetch_url"}:
                        console.print(f"[gray50]• {fn_name}...[/gray50]", end="\r")

                    handler = TOOL_DISPATCH.get(fn_name)
                    if handler is None:
                        result = f"ERROR: UnknownTool - No handler registered for '{fn_name}'."
                    else:
                        result = handler(fn_args)

                    if result.startswith("ERROR:"):
                        error_counts[fn_name] = error_counts.get(fn_name, 0) + 1
                        if fn_name in {"read_file", "glob_files", "grep_files", "get_job_output",
                                       "git_status", "git_diff", "git_log", "fetch_url"}:
                            console.print(" " * 40, end="\r")  # clear the running line
                        console.print(f"[dodger_blue1]•[/dodger_blue1] [dim white]{fn_name}[/dim white] [bright_red]✘[/bright_red]")
                        console.print(
                            Panel(
                                f"  {markup_escape(result)}",
                                title="[color(196)]✘ Execution Failed[/color(196)]",
                                border_style="color(196)",
                                box=box.SQUARE,
                            )
                        )
                        if error_counts[fn_name] >= 3:
                            console.print(
                                "[bold yellow]Agent stuck in error loop. Please intervene.[/bold yellow]"
                            )
                            stuck = True
                            break
                    else:
                        if fn_name in _SILENT_TOOLS:
                            console.print(f"[dodger_blue1]•[/dodger_blue1] [dim white]{fn_name}[/dim white] [bright_green]✔[/bright_green]" + " " * 20)
                        elif fn_name == "execute_bash":
                            console.print(f"[dodger_blue1]•[/dodger_blue1] [dim white]{fn_name}[/dim white] [bright_green]✔[/bright_green]")

                    function_response_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=fn_name,
                                response={"content": result},
                            )
                        )
                    )

                if stuck:
                    # The model's unanswered function-call turn is already in
                    # chat_history. Remove it so the next user message doesn't
                    # produce an invalid role sequence (model:fn_call → user:text).
                    chat_history.pop()
                    break

                chat_history.append(
                    types.Content(role="user", parts=function_response_parts)
                )

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Agent shutting down. Goodbye![/bold yellow]")
        sys.exit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Vertex AI Agentic REPL")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate write_file and execute_bash without touching disk or running commands.",
    )
    args = parser.parse_args()

    global DRY_RUN
    DRY_RUN = args.dry_run
    if DRY_RUN:
        console.print(
            "[bold yellow][DRY RUN] Mode active — "
            "writes and shell commands will be simulated.[/bold yellow]"
        )

    client, project_id, vagent_content = init_vertex()
    run_agent(client, project_id, vagent_content)


if __name__ == "__main__":
    main()
