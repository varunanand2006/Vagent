import argparse
import difflib
import os
import re
import subprocess
import sys
import time
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
_SYSTEM_INSTRUCTION: str | None = None


def _confirm_execution(prompt: str) -> bool:
    response = console.input(f"{prompt} — Allow execution? [bold dodger_blue1]\\[y/n][/bold dodger_blue1]: ").strip().lower()
    return response in ("y", "yes")


def read_file(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
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

local_tools = types.Tool(
    function_declarations=[
        _read_file_declaration,
        _write_file_declaration,
        _execute_bash_declaration,
        _list_directory_declaration,
    ]
)

TOOL_DISPATCH = {
    "read_file": lambda args: read_file(**args),
    "write_file": lambda args: write_file(**args),
    "execute_bash": lambda args: execute_bash(**args),
    "list_directory": lambda args: list_directory(**args),
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

    # Load project-specific context from .vagent in the current directory.
    vagent_content = ""
    vagent_path = Path(".vagent")
    if vagent_path.exists():
        try:
            vagent_content = vagent_path.read_text(encoding="utf-8").strip()
            if vagent_content:
                _SYSTEM_INSTRUCTION = f"Project Specific Context:\n{vagent_content}"
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
                try:
                    with console.status("[bold medium_purple1]Agent is thinking...[/bold medium_purple1]", spinner="dots"):
                        response = client.models.generate_content(
                            model=MODEL_NAME,
                            contents=chat_history,
                            config=types.GenerateContentConfig(
                                system_instruction=_SYSTEM_INSTRUCTION,
                                tools=[local_tools],
                            ),
                        )
                except Exception as api_err:
                    chat_history.pop()  # remove the user turn we just appended
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
                _SILENT_TOOLS = {"read_file", "list_directory"}
                function_response_parts: list[types.Part] = []
                stuck = False
                for fc in function_calls:
                    fn_name = fc.name
                    fn_args = dict(fc.args)

                    if fn_name == "write_file":
                        fp_display = fn_args.get("filepath", "?")
                        console.print(f"\n[bold white on dodger_blue1] ✎ EDIT [/bold white on dodger_blue1] [bold white] {fp_display} [/bold white]")
                    elif fn_name == "execute_bash":
                        cmd_display = fn_args.get("command", "?")
                        console.print(f"\n[bold white on medium_purple1] ❯ RUN [/bold white on medium_purple1] [bold white] {cmd_display} [/bold white]")
                    elif fn_name == "read_file":
                        console.print(f"[gray50]• read_file...[/gray50]", end="\r")

                    handler = TOOL_DISPATCH.get(fn_name)
                    if handler is None:
                        result = f"ERROR: UnknownTool - No handler registered for '{fn_name}'."
                    else:
                        result = handler(fn_args)

                    if result.startswith("ERROR:"):
                        error_counts[fn_name] = error_counts.get(fn_name, 0) + 1
                        if fn_name == "read_file":
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
