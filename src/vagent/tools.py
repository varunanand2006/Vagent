import difflib
import html as html_mod
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from google.genai import types
from rich import box
from rich.console import Console
from rich.markup import escape as markup_escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()
DRY_RUN: bool = False
_confirm_callback = None
_read_files: set[str] = set()
_background_jobs: dict[str, dict] = {}
_job_counter: int = 0
_change_records: list[dict] = []


def init_tools(custom_console, confirm_cb, dry_run=False, pre_read=None):
    global console, _confirm_callback, DRY_RUN, _read_files
    console = custom_console
    _confirm_callback = confirm_cb
    DRY_RUN = dry_run
    if pre_read:
        _read_files.update(str(Path(p).resolve()) for p in pre_read)


def get_read_files():
    return _read_files


def clear_read_files():
    _read_files.clear()


def get_change_records():
    return _change_records


def _record(kind: str, detail: str, outcome: str):
    _change_records.append({"kind": kind, "detail": detail, "outcome": outcome})


def _confirm_execution(prompt: str) -> bool:
    if DRY_RUN:
        console.print(f"[dim][DRY RUN] {prompt}[/dim]")
        return True
    if _confirm_callback:
        return _confirm_callback(prompt)
    return True


def cleanup_background_jobs():
    for job_id, job in list(_background_jobs.items()):
        proc = job["process"]
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
    _background_jobs.clear()


def _render_file_diff(diff_lines: list[str], filepath: str, is_new_file: bool, added: int, removed: int) -> None:
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

    hunks: list[dict] = []
    current_hunk: dict | None = None
    for raw in diff_lines:
        if raw.startswith("---") or raw.startswith("+++"):
            continue
        m = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
        if m:
            current_hunk = {"old_start": int(m.group(1)), "new_start": int(m.group(2)), "lines": []}
            hunks.append(current_hunk)
        elif current_hunk is not None and not raw.startswith("\\"):
            current_hunk["lines"].append(raw)

    if not hunks:
        return

    max_lineno = max(h["new_start"] + sum(1 for ln in h["lines"] if not ln.startswith("-")) for h in hunks)
    gutter_w = max(len(str(max_lineno)), 3)

    MAX_CHANGED = 80
    total_changed = 0
    truncated = False

    for hunk_idx, hunk in enumerate(hunks):
        if hunk_idx > 0:
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
        console.print(f"  [dim]… {remaining} more change{'s' if remaining != 1 else ''} not shown[/dim]")


class _TextExtractor(HTMLParser):
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
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------

def read_file(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        _read_files.add(str(Path(filepath).resolve()))
        return content
    except Exception as e:
        return f"ERROR: {type(e).__name__} - {e}"


def write_file(filepath: str, content: str) -> str:
    fp = Path(filepath)
    is_new_file = not fp.exists()
    old_content = ""
    if not is_new_file:
        if str(fp.resolve()) not in _read_files:
            return (
                f"ERROR: ReadRequired - '{filepath}' already exists but has not been read. "
                "Call read_file first."
            )
        try:
            old_content = fp.read_text(encoding="utf-8")
        except Exception:
            pass

    if not _confirm_execution(f"[bold #ffffd7]write_file({filepath!r})[/bold #ffffd7]"):
        _record("write_file", filepath, "blocked")
        return "Execution blocked by user."

    if DRY_RUN:
        _record("write_file", filepath, "dry-run")
        return "Success"

    try:
        fp.write_text(content, encoding="utf-8")
    except Exception as e:
        _record("write_file", filepath, f"error:{e}")
        return f"ERROR: {type(e).__name__} - {e}"

    _record("write_file", filepath, "success")
    old_lines = old_content.splitlines(keepends=True)
    new_lines = content.splitlines(keepends=True)
    diff_lines = list(difflib.unified_diff(
        old_lines, new_lines, fromfile=f"a/{filepath}", tofile=f"b/{filepath}", lineterm="", n=3,
    ))
    added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
    _render_file_diff(diff_lines, filepath, is_new_file, added, removed)
    return f"Successfully wrote {len(content)} bytes to {filepath!r}."


def edit_file(filepath: str, old_string: str, new_string: str) -> str:
    fp = Path(filepath)
    if not fp.exists():
        return f"ERROR: FileNotFound - '{filepath}' does not exist."
    if str(fp.resolve()) not in _read_files:
        return f"ERROR: ReadRequired - '{filepath}' must be read before editing."

    try:
        content = fp.read_text(encoding="utf-8")
    except Exception as e:
        return f"ERROR: {type(e).__name__} - {e}"

    count = content.count(old_string)
    if count == 0:
        return f"ERROR: StringNotFound - Exact string not found in '{filepath}'."
    if count > 1:
        return f"ERROR: AmbiguousMatch - String appears {count} times."

    if not _confirm_execution(f"[bold #ffffd7]edit_file({filepath!r})[/bold #ffffd7]"):
        _record("edit_file", filepath, "blocked")
        return "Execution blocked by user."

    if DRY_RUN:
        _record("edit_file", filepath, "dry-run")
        return "Success"

    new_content = content.replace(old_string, new_string, 1)
    try:
        fp.write_text(new_content, encoding="utf-8")
    except Exception as e:
        _record("edit_file", filepath, f"error:{e}")
        return f"ERROR: {type(e).__name__} - {e}"

    _record("edit_file", filepath, "success")
    old_lines = content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff_lines = list(difflib.unified_diff(
        old_lines, new_lines, fromfile=f"a/{filepath}", tofile=f"b/{filepath}", lineterm="", n=3,
    ))
    added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
    _render_file_diff(diff_lines, filepath, False, added, removed)
    return f"Edited '{filepath}' ({added} added, {removed} removed)."


def save_memory(fact: str) -> str:
    if not _confirm_execution(f"[bold deep_pink3]save_memory({fact!r})[/bold deep_pink3]"):
        _record("save_memory", fact, "blocked")
        return "Execution blocked by user."

    if DRY_RUN:
        _record("save_memory", fact, "dry-run")
        return "Success"
        
    try:
        vagent_path = Path(".vagent")
        existing = ""
        if vagent_path.exists():
            existing = vagent_path.read_text(encoding="utf-8")
            
        if fact in existing:
            _record("save_memory", fact, "success")
            return "Fact already remembered."
            
        with open(vagent_path, "a", encoding="utf-8") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write(f"- {fact}\n")
            
        _record("save_memory", fact, "success")
        return "Successfully saved fact to memory."
    except Exception as e:
        _record("save_memory", fact, f"error:{e}")
        return f"ERROR: {type(e).__name__} - {e}"


def ask_user(question: str) -> str:
    if DRY_RUN:
        console.print(f"[dim][DRY RUN] Would have asked user: {question!r}[/dim]")
        return "Success (Dry Run Response)"
        
    console.print()
    console.print(Panel(
        f"[bold white]{markup_escape(question)}[/bold white]",
        title="[bold dodger_blue1]Agent Question[/bold dodger_blue1]",
        border_style="dodger_blue1"
    ))
    
    response = console.input(f"[bold dodger_blue1]❯[/bold dodger_blue1] ").strip()
    return f"User response: {response}"


def list_directory(path: str = ".") -> str:
    _ICON_MAP = {".py": "🐍", ".txt": "📄", ".json": "⚙️", ".md": "📝"}
    _KIND_MAP = {".py": "Python", ".txt": "Text", ".json": "JSON", ".md": "Markdown"}

    try:
        entries = sorted(os.listdir(path))
    except Exception as e:
        return f"ERROR: {type(e).__name__} - {e}"

    if not entries:
        return f"(empty directory: {path!r})"

    tbl = Table(border_style="dodger_blue1", show_header=True, header_style="bold gray93", box=box.SIMPLE_HEAD)
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
        sz = "—" if is_dir else f"{os.path.getsize(full)} B" if os.path.isfile(full) else "?"
        name_display = f"[bold medium_purple1]{name}[/bold medium_purple1]" if is_dir else f"[white]{name}[/white]"
        tbl.add_row(icon, name_display, sz, kind)

    console.print(tbl)
    return "\n".join(entries)


def execute_bash(command: str) -> str:
    if not _confirm_execution(f"[bold #ffffd7]execute_bash({command!r})[/bold #ffffd7]"):
        _record("execute_bash", command, "blocked")
        return "Execution blocked by user."
    if DRY_RUN:
        _record("execute_bash", command, "dry-run")
        return "Success"
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            _record("execute_bash", command, f"error:exit={result.returncode}")
            return f"ERROR: NonZeroExit - code {result.returncode}.\nSTDERR: {result.stderr.strip()}"
        _record("execute_bash", command, "success")
        return result.stdout or "(no output)"
    except Exception as e:
        _record("execute_bash", command, f"error:{e}")
        return f"ERROR: {type(e).__name__} - {e}"


def execute_bash_background(command: str) -> str:
    global _job_counter
    if not _confirm_execution(f"[bold #ffffd7]execute_bash_background({command!r})[/bold #ffffd7]"):
        _record("execute_bash_background", command, "blocked")
        return "Execution blocked by user."
    if DRY_RUN:
        _record("execute_bash_background", command, "dry-run")
        return "Success"

    _job_counter += 1
    job_id = str(_job_counter)
    stdout_path = os.path.join(tempfile.gettempdir(), f"vagent_job_{job_id}.out")
    stderr_path = os.path.join(tempfile.gettempdir(), f"vagent_job_{job_id}.err")

    try:
        stdout_f = open(stdout_path, "w", encoding="utf-8")
        stderr_f = open(stderr_path, "w", encoding="utf-8")
        process = subprocess.Popen(command, shell=True, stdout=stdout_f, stderr=stderr_f, text=True)
        _background_jobs[job_id] = {
            "process": process, "command": command, "start_time": time.monotonic(),
            "stdout_path": stdout_path, "stderr_path": stderr_path,
            "stdout_f": stdout_f, "stderr_f": stderr_f,
        }
        _record("execute_bash_background", command, "success")
        return f"Job {job_id} started (PID {process.pid}). Use get_job_output('{job_id}')."
    except Exception as e:
        _record("execute_bash_background", command, f"error:{e}")
        return f"ERROR: {type(e).__name__} - {e}"


def get_job_output(job_id: str) -> str:
    job = _background_jobs.get(job_id)
    if job is None:
        return f"ERROR: No job with ID '{job_id}'."

    process = job["process"]
    returncode = process.poll()
    elapsed = time.monotonic() - job["start_time"]

    if returncode is not None:
        try:
            job["stdout_f"].close()
            job["stderr_f"].close()
        except Exception: pass

    try:
        stdout = Path(job["stdout_path"]).read_text(encoding="utf-8", errors="ignore")
        stderr = Path(job["stderr_path"]).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        stdout, stderr = "", ""

    if returncode is None:
        status = f"RUNNING ({elapsed:.1f}s elapsed)"
    else:
        status = f"DONE (exit code {returncode}, {elapsed:.1f}s)"
        try:
            os.unlink(job["stdout_path"])
            os.unlink(job["stderr_path"])
        except Exception: pass
        del _background_jobs[job_id]

    lines = [f"Job {job_id}: {job['command']!r}", f"Status: {status}"]
    if stdout.strip(): lines.append(f"STDOUT:\n{stdout.rstrip()}")
    if stderr.strip(): lines.append(f"STDERR:\n{stderr.rstrip()}")
    return "\n".join(lines)


def delegate_task(task: str, context: str = "") -> str:
    if not _confirm_execution(f"[bold medium_purple1]delegate_task({task!r})[/bold medium_purple1]"):
        _record("delegate_task", task, "blocked")
        return "Execution blocked by user."
    if DRY_RUN:
        _record("delegate_task", task, "dry-run")
        return "Success (Dry Run)"

    console.print(f"[bold medium_purple1]Spawning Subagent for task:[/bold medium_purple1] {task}")

    enhanced_context = ""
    if _read_files:
        enhanced_context += f"Files recently analyzed by main agent: {', '.join(_read_files)}\n\n"
    if context:
        enhanced_context += context

    subagent_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subagent.py")
    cmd = [sys.executable, subagent_script, task, "--json", "--auto-confirm"]
    if enhanced_context.strip():
        cmd.extend(["--context", enhanced_context.strip()])
    
    # Forward the read files state
    for f in _read_files:
        cmd.extend(["--pre-read", f])

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        if result.returncode != 0 and not result.stdout.strip():
            _record("delegate_task", task, f"error:exit={result.returncode}")
            return f"ERROR: Subagent failed (exit {result.returncode})."

        import json
        try:
            data = json.loads(result.stdout.strip())
            out = [f"Subagent Task: {task}", f"Success: {data.get('success')}"]
            if data.get('error'): out.append(f"Error: {data.get('error')}")
            out.append(f"\nSummary:\n{data.get('summary', 'No summary.')}")
            changes = data.get('changes', [])
            if changes:
                out.append("\nChanges made:")
                for c in changes:
                    out.append(f"- {c.get('kind')}: {c.get('detail')} ({c.get('outcome')})")
            _record("delegate_task", task, "success")
            return "\n".join(out)
        except json.JSONDecodeError:
            _record("delegate_task", task, "error:invalid_json")
            return f"ERROR: Invalid JSON from subagent.\nSTDOUT: {result.stdout}"
    except Exception as e:
        _record("delegate_task", task, f"error:{e}")
        return f"ERROR: Failed to launch subagent: {e}"


def git_status() -> str:
    try:
        r = subprocess.run(["git", "status", "--short", "--branch"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "(clean working tree)" if r.returncode == 0 else f"ERROR: {r.stderr.strip()}"
    except Exception as e: return f"ERROR: {type(e).__name__} - {e}"


def git_diff(filepath: str = "") -> str:
    try:
        cmd = ["git", "diff"] + ([filepath] if filepath else [])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        output = r.stdout.strip()
        if len(output) > 8000: output = output[:8000] + "\n...(truncated)"
        return output or "(no unstaged changes)"
    except Exception as e: return f"ERROR: {type(e).__name__} - {e}"


def git_log(max_entries: int = 10) -> str:
    try:
        r = subprocess.run(["git", "log", f"--max-count={max_entries}", "--oneline", "--decorate"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "(no commits)"
    except Exception as e: return f"ERROR: {type(e).__name__} - {e}"


def git_add(files: list = None) -> str:
    if not _confirm_execution(f"[bold #ffffd7]git_add({', '.join(files) if files else 'all'})[/bold #ffffd7]"):
        _record("git_add", str(files or "all"), "blocked")
        return "Execution blocked by user."
    if DRY_RUN:
        _record("git_add", str(files or "all"), "dry-run")
        return "Success"
    try:
        cmd = ["git", "add"] + (files if files else ["-A"])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            _record("git_add", str(files), f"error:{r.stderr.strip()}")
            return f"ERROR: {r.stderr.strip()}"
        _record("git_add", str(files or "all"), "success")
        return r.stdout.strip() or "Staged successfully."
    except Exception as e: return f"ERROR: {type(e).__name__} - {e}"


def git_commit(message: str, files: list = None) -> str:
    if not _confirm_execution(f"[bold #ffffd7]git_commit({message!r})[/bold #ffffd7]"):
        _record("git_commit", message, "blocked")
        return "Execution blocked by user."
    if DRY_RUN:
        _record("git_commit", message, "dry-run")
        return "Success"
    try:
        subprocess.run(["git", "add"] + (files if files else ["-A"]), capture_output=True, text=True, timeout=10)
        r = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            _record("git_commit", message, f"error:{r.stderr.strip()}")
            return f"ERROR: git commit failed: {r.stderr.strip()}"
        _record("git_commit", message, "success")
        return r.stdout.strip()
    except Exception as e: return f"ERROR: {type(e).__name__} - {e}"


def glob_files(pattern: str, path: str = ".") -> str:
    try:
        matches = sorted(str(p) for p in Path(path).glob(pattern))
        if not matches: return f"No files matched {pattern!r}"
        return "\n".join(matches[:500]) + ("\n...(truncated)" if len(matches) > 500 else "")
    except Exception as e: return f"ERROR: {type(e).__name__} - {e}"


def grep_files(pattern: str, path: str = ".", glob: str = "**/*") -> str:
    try: regex = re.compile(pattern)
    except re.error as e: return f"ERROR: Invalid regex - {e}"
    results = []
    try:
        for fp in sorted(Path(path).glob(glob)):
            if not fp.is_file(): continue
            try: lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception: continue
            for i, line in enumerate(lines, 1):
                if regex.search(line):
                    results.append(f"{fp}:{i}: {line}")
                    if len(results) >= 200: break
            if len(results) >= 200: break
        return "\n".join(results) if results else f"No matches for {pattern!r}"
    except Exception as e: return f"ERROR: {type(e).__name__} - {e}"


def fetch_url(url: str) -> str:
    MAX_CHARS = 12_000
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(MAX_CHARS * 4).decode("utf-8", errors="replace")
        if "text/html" in content_type:
            parser = _TextExtractor()
            parser.feed(html_mod.unescape(raw))
            raw = parser.get_text()
        return raw[:MAX_CHARS] + ("\n...(truncated)" if len(raw) > MAX_CHARS else "")
    except Exception as e: return f"ERROR: {type(e).__name__} - {e}"


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

S = types.Schema
T = types.Type
def prop(**kw): return S(**kw)

local_tools = types.Tool(function_declarations=[
    types.FunctionDeclaration(name="read_file", description="Read the full text of a local file.", parameters=S(type=T.OBJECT, properties={"filepath": prop(type=T.STRING, description="Path to the file.")}, required=["filepath"])),
    types.FunctionDeclaration(name="write_file", description="Write (overwrite) a file.", parameters=S(type=T.OBJECT, properties={"filepath": prop(type=T.STRING), "content": prop(type=T.STRING)}, required=["filepath", "content"])),
    types.FunctionDeclaration(name="edit_file", description="Replace exact string in file. Must read first.", parameters=S(type=T.OBJECT, properties={"filepath": prop(type=T.STRING), "old_string": prop(type=T.STRING), "new_string": prop(type=T.STRING)}, required=["filepath", "old_string", "new_string"])),
    types.FunctionDeclaration(name="execute_bash", description="Run a shell command.", parameters=S(type=T.OBJECT, properties={"command": prop(type=T.STRING)}, required=["command"])),
    types.FunctionDeclaration(name="execute_bash_background", description="Start background command. Returns job ID.", parameters=S(type=T.OBJECT, properties={"command": prop(type=T.STRING)}, required=["command"])),
    types.FunctionDeclaration(name="get_job_output", description="Poll a background job.", parameters=S(type=T.OBJECT, properties={"job_id": prop(type=T.STRING)}, required=["job_id"])),
    types.FunctionDeclaration(name="delegate_task", description="Spawn subagent for complex task.", parameters=S(type=T.OBJECT, properties={"task": prop(type=T.STRING), "context": prop(type=T.STRING)}, required=["task"])),
    types.FunctionDeclaration(name="save_memory", description="Save fact across sessions in .vagent.", parameters=S(type=T.OBJECT, properties={"fact": prop(type=T.STRING)}, required=["fact"])),
    types.FunctionDeclaration(name="ask_user", description="Ask user a question and pause loop.", parameters=S(type=T.OBJECT, properties={"question": prop(type=T.STRING)}, required=["question"])),
    types.FunctionDeclaration(name="list_directory", description="List directory contents.", parameters=S(type=T.OBJECT, properties={"path": prop(type=T.STRING)})),
    types.FunctionDeclaration(name="glob_files", description="Find files via glob pattern.", parameters=S(type=T.OBJECT, properties={"pattern": prop(type=T.STRING), "path": prop(type=T.STRING)}, required=["pattern"])),
    types.FunctionDeclaration(name="grep_files", description="Regex search files.", parameters=S(type=T.OBJECT, properties={"pattern": prop(type=T.STRING), "path": prop(type=T.STRING), "glob": prop(type=T.STRING)}, required=["pattern"])),
    types.FunctionDeclaration(name="fetch_url", description="Fetch URL text content.", parameters=S(type=T.OBJECT, properties={"url": prop(type=T.STRING)}, required=["url"])),
    types.FunctionDeclaration(name="git_status", description="Git status.", parameters=S(type=T.OBJECT, properties={})),
    types.FunctionDeclaration(name="git_diff", description="Git diff.", parameters=S(type=T.OBJECT, properties={"filepath": prop(type=T.STRING)})),
    types.FunctionDeclaration(name="git_log", description="Git log.", parameters=S(type=T.OBJECT, properties={"max_entries": prop(type=T.INTEGER)})),
    types.FunctionDeclaration(name="git_add", description="Git add.", parameters=S(type=T.OBJECT, properties={"files": S(type=T.ARRAY, items=S(type=T.STRING))})),
    types.FunctionDeclaration(name="git_commit", description="Git commit.", parameters=S(type=T.OBJECT, properties={"message": prop(type=T.STRING), "files": S(type=T.ARRAY, items=S(type=T.STRING))}, required=["message"]))
])

TOOL_DISPATCH = {
    "read_file": lambda args: read_file(**args),
    "write_file": lambda args: write_file(**args),
    "edit_file": lambda args: edit_file(**args),
    "execute_bash": lambda args: execute_bash(**args),
    "execute_bash_background": lambda args: execute_bash_background(**args),
    "get_job_output": lambda args: get_job_output(**args),
    "delegate_task": lambda args: delegate_task(**args),
    "save_memory": lambda args: save_memory(**args),
    "ask_user": lambda args: ask_user(**args),
    "list_directory": lambda args: list_directory(**args),
    "glob_files": lambda args: glob_files(**args),
    "grep_files": lambda args: grep_files(**args),
    "fetch_url": lambda args: fetch_url(**args),
    "git_status": lambda args: git_status(**args),
    "git_diff": lambda args: git_diff(**args),
    "git_log": lambda args: git_log(**args),
    "git_add": lambda args: git_add(**args),
    "git_commit": lambda args: git_commit(**args)
}
