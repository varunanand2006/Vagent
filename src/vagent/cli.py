import argparse
import os
import re
import sys
import threading
import time
import mimetypes
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

from vagent import tools

console = Console()
DRY_RUN: bool = False
PLAN_MODE: bool = False
MODEL_NAME = "gemini-2.5-pro"
LOCATION = "us-central1"
TEMPERATURE = 0.3
_SYSTEM_INSTRUCTION: str | None = None

_PLAN_MODE_BLOCKED_TOOLS = {
    "write_file", "edit_file", "execute_bash", "execute_bash_background",
    "git_add", "git_commit",
}
_PLAN_MODE_ADDENDUM = """\

## PLAN MODE ACTIVE
You are currently in plan mode. Do NOT call any of these tools: \
write_file, edit_file, execute_bash, execute_bash_background, git_add, git_commit.
You MAY use read-only tools (read_file, glob_files, grep_files, list_directory, \
git_status, git_diff, git_log, fetch_url) to explore the codebase.
Instead of executing changes, describe exactly what you would do: which files you \
would modify, what the diff would look like, and why. Be specific and actionable.
"""

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
- Use save_memory to persist important project rules, user preferences, or discovered architectural patterns.

## Behaviour
- Make the smallest correct change. Do not refactor or reformat code outside the \
scope of the request.
- When a task requires multiple steps, state your plan briefly before executing it.
- **Autonomous Verification:** After making code changes, ALWAYS use execute_bash to run the relevant build, lint, or test commands to verify your changes. If tests fail, autonomously diagnose and fix the errors before concluding your response.
- If something is unclear, ask a single focused question rather than proceeding \
on assumptions.
- Never delete or overwrite files without explicit instruction to do so.
"""

def _confirm_execution(prompt: str) -> bool:
    response = console.input(f"{prompt} — Allow execution? [bold dodger_blue1]\\[y/n][/bold dodger_blue1]: ").strip().lower()
    return response in ("y", "yes")

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

    console.print(f"[gray50]Compacted {len(chat_history)} → {len(compacted)} turns.[/gray50]")
    return compacted

# ---------------------------------------------------------------------------
# Response rendering
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

def render_response(text: str) -> None:
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

def print_environment_header(vagent_content: str) -> None:
    tool_count = len(tools.local_tools.function_declarations)
    if DRY_RUN: mode = "[bold yellow]DRY RUN[/bold yellow]"
    elif PLAN_MODE: mode = "[bold orange1]Plan[/bold orange1]"
    else: mode = "[bold dodger_blue1]Normal[/bold dodger_blue1]"
    vagent_status = "[bold dodger_blue1]Loaded[/bold dodger_blue1]" if vagent_content else "[gray50]None[/gray50]"

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column()
    grid.add_row("Current Path:", os.getcwd())
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
    cmd_table.add_row("[medium_purple1]/plan[/medium_purple1]", "Toggle plan mode — model describes changes without executing them")
    cmd_table.add_row("[medium_purple1]/help[/medium_purple1]  or  [medium_purple1]?[/medium_purple1]", "Show this help")

    tools_table = Table(border_style="dim", show_header=True, header_style="bold")
    tools_table.add_column("Tool", style="bold #0087ff", no_wrap=True)
    tools_table.add_column("Safety Gate")
    tools_table.add_column("Description")
    for decl in tools.local_tools.function_declarations:
        if decl.name in {"read_file", "list_directory"}: gate = "[dim]None (read-only)[/dim]"
        else: gate = "[yellow]Confirm [Y/n][/yellow]"
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
        key_bindings=kb, multiline=True,
        prompt_continuation=lambda width, line_number, is_soft_wrap: "... ",
    )


def _generate_with_cancel(client: genai.Client, contents: list, config: types.GenerateContentConfig) -> tuple:
    result: list = [None]
    exc: list = [None]
    done = threading.Event()

    def _worker():
        try: result[0] = client.models.generate_content(model=MODEL_NAME, contents=contents, config=config)
        except Exception as e: exc[0] = e
        finally: done.set()

    threading.Thread(target=_worker, daemon=True).start()

    try:
        import msvcrt
        def _escape_pressed() -> bool: return msvcrt.kbhit() and msvcrt.getch() == b"\x1b"
    except ImportError:
        import select, tty, termios
        def _escape_pressed() -> bool:
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                return bool(select.select([sys.stdin], [], [], 0)[0]) and sys.stdin.read(1) == "\x1b"
            except Exception: return False
            finally: termios.tcsetattr(fd, termios.TCSADRAIN, old)

    while not done.wait(timeout=0.05):
        if _escape_pressed(): return None, True

    if exc[0]: raise exc[0]
    return result[0], False


def _parse_user_input_for_images(user_text: str) -> list[types.Part]:
    parts = []
    image_exts = r"\.(png|jpg|jpeg|webp|heic|gif)$"
    path_pattern = r"(?P<path>(?:[a-zA-Z]:\\|/|\./|\.\./|~/?)[^\s\"']+?" + image_exts + r")"
    
    found_paths = set()
    for match in re.finditer(path_pattern, user_text, re.IGNORECASE):
        path_str = match.group("path")
        if path_str in found_paths: continue
        clean_path = path_str.strip("'\"")
        try:
            expanded_path = os.path.expanduser(clean_path)
            if os.path.exists(expanded_path) and os.path.isfile(expanded_path):
                mime_type, _ = mimetypes.guess_type(expanded_path)
                with open(expanded_path, "rb") as f:
                    image_bytes = f.read()
                parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type or "image/png"))
                found_paths.add(path_str)
                console.print(f"[dim dodger_blue1]Attached image:[/dim dodger_blue1] [white]{clean_path}[/white]")
        except Exception as e:
            console.print(f"[dim red]Warning: Found image path {clean_path!r} but failed to load it: {e}[/dim red]")

    parts.append(types.Part.from_text(user_text))
    return parts

def run_agent(client: genai.Client, project_id: str, vagent_content: str) -> None:
    chat_history: list[types.Content] = []
    print_environment_header(vagent_content)
    prompt_session = _build_prompt_session()

    try:
        while True:
            try:
                console.print()
                console.print(Rule(style="dim medium_purple1"))
                user_text = prompt_session.prompt(HTML("<style fg='#0087ff'><b>❯ </b></style>")).strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye.[/]")
                break

            if not user_text: continue

            if user_text in {"?", "/help"} or user_text.startswith("/"):
                if user_text in {"?", "/help"}: print_help()
                elif user_text == "/exit":
                    console.print("[dim]Goodbye.[/]")
                    sys.exit(0)
                elif user_text == "/clear":
                    if not chat_history:
                        console.print("[dim]History is already empty.[/dim]")
                    else:
                        chat_history.clear()
                        tools.clear_read_files()
                        console.print("[bold red]History cleared.[/bold red]")
                elif user_text == "/compact":
                    chat_history = compact_history(chat_history, client)
                elif user_text == "/plan":
                    global PLAN_MODE
                    PLAN_MODE = not PLAN_MODE
                    if PLAN_MODE: console.print("[bold orange1]Plan mode ON[/bold orange1] — model will describe changes without executing them.")
                    else: console.print("[bold dodger_blue1]Plan mode OFF[/bold dodger_blue1] — normal execution resumed.")
                else:
                    console.print(f"[gray50]Unknown command: {user_text!r}. Available: /exit, /clear, /compact, /plan[/gray50]")
                continue

            input_parts = _parse_user_input_for_images(user_text)
            chat_history.append(types.Content(role="user", parts=input_parts))

            turn_start = time.monotonic()
            error_counts: dict[str, int] = {}
            while True:
                effective_system = ((_SYSTEM_INSTRUCTION or "") + _PLAN_MODE_ADDENDUM) if PLAN_MODE else _SYSTEM_INSTRUCTION
                config = types.GenerateContentConfig(
                    system_instruction=effective_system,
                    tools=[tools.local_tools, types.Tool(google_search=types.GoogleSearch())],
                    temperature=TEMPERATURE,
                )
                status = console.status("[bold medium_purple1]Agent is thinking...[/bold medium_purple1]  [dim](Esc to cancel)[/dim]", spinner="dots")
                status.start()
                try:
                    response, cancelled = _generate_with_cancel(client, chat_history, config)
                except Exception as api_err:
                    status.stop()
                    chat_history.pop()
                    err_str = str(api_err)
                    console.print(Panel(f"[bold]{markup_escape(err_str.split('{')[0].strip().rstrip('.'))}[/bold]\n\n{markup_escape(getattr(api_err, 'message', '') or '')}", title="[bold red]✘ API Error[/bold red]", border_style="red"))
                    break
                finally:
                    status.stop()

                if cancelled:
                    chat_history.pop()
                    console.print("[dim]Cancelled.[/dim]")
                    break

                candidate = response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    console.print(Panel("The model returned an empty or blocked response.", title="[bold red]✘ Empty Response[/bold red]", border_style="red"))
                    break

                chat_history.append(candidate.content)

                function_calls = [p.function_call for p in candidate.content.parts if p.function_call and p.function_call.name]

                if not function_calls:
                    text = "\n".join(p.text for p in candidate.content.parts if p.text)
                    if not text:
                        console.print("[dim]Model returned no text.[/dim]")
                        break
                    console.print()
                    console.print(Rule("[italic gray50]Gemini[/italic gray50]", style="dim medium_purple1"))
                    console.print()
                    render_response(text)
                    console.print()

                    grounding = getattr(candidate, "grounding_metadata", None)
                    if grounding and (chunks := getattr(grounding, "grounding_chunks", [])):
                        console.print("[gray50]Sources:[/gray50]")
                        for chunk in chunks:
                            if web := getattr(chunk, "web", None):
                                console.print(f"  [gray50]• [link={getattr(web, 'uri', '')}]{markup_escape(getattr(web, 'title', '') or getattr(web, 'uri', ''))}[/link][/gray50]")
                        console.print()

                    total_tokens = response.usage_metadata.total_token_count
                    turn_time = time.monotonic() - turn_start
                    console.print(f"[gray50]✦ Calculated... ({turn_time:.1f}s • {total_tokens:,} tokens)[/gray50]")

                    if total_tokens > COMPACT_TOKEN_THRESHOLD:
                        console.print(f"[dim yellow]Token threshold exceeded ({total_tokens:,} > {COMPACT_TOKEN_THRESHOLD:,}). Auto-compacting...[/dim yellow]")
                        chat_history = compact_history(chat_history, client)
                    break

                _SILENT_TOOLS = {"read_file", "list_directory", "glob_files", "grep_files", "get_job_output", "fetch_url", "git_status", "git_diff", "git_log"}
                function_response_parts: list[types.Part] = []
                stuck = False
                for fc in function_calls:
                    fn_name = fc.name
                    fn_args = dict(fc.args)

                    if fn_name == "write_file": console.print(f"\n[bold white on dodger_blue1] ✎ WRITE [/bold white on dodger_blue1] [bold white] {fn_args.get('filepath', '?')} [/bold white]")
                    elif fn_name == "edit_file": console.print(f"\n[bold white on dodger_blue1] ✎ EDIT [/bold white on dodger_blue1] [bold white] {fn_args.get('filepath', '?')} [/bold white]")
                    elif fn_name == "git_add": console.print(f"\n[bold white on green4] ⎇ ADD [/bold white on green4] [bold white] {', '.join(fn_args.get('files') or ['all'])} [/bold white]")
                    elif fn_name == "git_commit": console.print(f"\n[bold white on green4] ⎇ COMMIT [/bold white on green4] [bold white] {fn_args.get('message', '?')} [/bold white]")
                    elif fn_name == "execute_bash": console.print(f"\n[bold white on medium_purple1] ❯ RUN [/bold white on medium_purple1] [bold white] {fn_args.get('command', '?')} [/bold white]")
                    elif fn_name == "execute_bash_background": console.print(f"\n[bold white on gray30] ❯ BG [/bold white on gray30] [bold white] {fn_args.get('command', '?')} [/bold white]")
                    elif fn_name == "save_memory": console.print(f"\n[bold white on deep_pink3] 🧠 REMEMBER [/bold white on deep_pink3] [bold white] {fn_args.get('fact', '?')} [/bold white]")
                    elif fn_name in _SILENT_TOOLS: console.print(f"[gray50]• {fn_name}...[/gray50]", end="\r")

                    handler = tools.TOOL_DISPATCH.get(fn_name)
                    if handler is None: result = f"ERROR: UnknownTool - No handler registered for '{fn_name}'."
                    elif PLAN_MODE and fn_name in _PLAN_MODE_BLOCKED_TOOLS: result = f"ERROR: PlanMode - '{fn_name}' is disabled in plan mode."
                    else: result = handler(fn_args)

                    if result.startswith("ERROR:"):
                        error_counts[fn_name] = error_counts.get(fn_name, 0) + 1
                        if fn_name in _SILENT_TOOLS: console.print(" " * 40, end="\r")
                        console.print(f"[dodger_blue1]•[/dodger_blue1] [dim white]{fn_name}[/dim white] [bright_red]✘[/bright_red]")
                        console.print(Panel(f"  {markup_escape(result)}", title="[color(196)]✘ Execution Failed[/color(196)]", border_style="color(196)", box=box.SQUARE))
                        if error_counts[fn_name] >= 3:
                            console.print("[bold yellow]Agent stuck in error loop. Please intervene.[/bold yellow]")
                            stuck = True
                            break
                    else:
                        if fn_name in _SILENT_TOOLS: console.print(f"[dodger_blue1]•[/dodger_blue1] [dim white]{fn_name}[/dim white] [bright_green]✔[/bright_green]" + " " * 20)
                        elif fn_name == "execute_bash": console.print(f"[dodger_blue1]•[/dodger_blue1] [dim white]{fn_name}[/dim white] [bright_green]✔[/bright_green]")

                    function_response_parts.append(types.Part(function_response=types.FunctionResponse(name=fn_name, response={"content": result})))

                if stuck:
                    chat_history.pop()
                    break

                chat_history.append(types.Content(role="user", parts=function_response_parts))

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Agent shutting down. Goodbye![/bold yellow]")
    finally:
        tools.cleanup_background_jobs()
        sys.exit(0)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Vertex AI Agentic REPL")
    parser.add_argument("--dry-run", action="store_true", help="Simulate writes and shell commands.")
    args = parser.parse_args()

    global DRY_RUN
    DRY_RUN = args.dry_run
    
    tools.init_tools(console, _confirm_execution, dry_run=DRY_RUN)

    if DRY_RUN:
        console.print("[bold yellow][DRY RUN] Mode active — writes and shell commands will be simulated.[/bold yellow]")

    client, project_id, vagent_content = init_vertex()
    run_agent(client, project_id, vagent_content)

if __name__ == "__main__":
    main()
