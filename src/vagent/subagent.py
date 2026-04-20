"""
gemini_subagent.py — Gemini as a subagent, supervised by Claude.

Claude describes a task; Gemini executes it autonomously using local tools
(read/write files, bash, git, etc.), then returns a structured summary of
every change it made.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from google import genai
from google.auth import default as google_auth_default
from google.genai import types
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape as markup_escape
from rich.panel import Panel
from rich.panel import Panel
from rich.table import Table

from vagent import tools

console = Console()

@dataclass
class ChangeRecord:
    kind: str
    detail: str
    outcome: str

@dataclass
class SubagentResult:
    task: str
    summary: str
    changes: list[ChangeRecord] = field(default_factory=list)
    success: bool = True
    error: str = ""

    def display(self) -> None:
        console.print()
        console.print(Rule("[bold medium_purple1]Gemini Subagent — Result[/bold medium_purple1]", style="dim medium_purple1"))

        if not self.success:
            console.print(Panel(markup_escape(self.error), title="[bold red]FAILED[/bold red]", border_style="red"))
            return

        if self.changes:
            tbl = Table(border_style="dim", show_header=True, header_style="bold", box=box.SIMPLE_HEAD)
            tbl.add_column("Action", style="dodger_blue1", no_wrap=True)
            tbl.add_column("Detail")
            tbl.add_column("Outcome", no_wrap=True)
            for c in self.changes:
                outcome_style = "bright_green" if c.outcome == "success" else "bright_red"
                tbl.add_row(c.kind, c.detail, f"[{outcome_style}]{c.outcome}[/{outcome_style}]")
            console.print(Panel(tbl, title="[bold]Changes Made[/bold]", border_style="dodger_blue1"))

        console.print()
        console.print("[bold]Gemini's Summary:[/bold]")
        console.print(Markdown(self.summary))
        console.print()


_SUBAGENT_SYSTEM_PROMPT = """\
You are a Gemini coding subagent. You have been assigned a specific task by
Claude (your supervisor). Complete it fully and accurately using the tools
available to you.

## Rules
- Always call glob_files or list_directory first to understand the project layout.
- Always call read_file before editing or overwriting any existing file.
- Prefer edit_file over write_file for changes to existing files.
- Make the minimal correct change — do not refactor beyond the task scope.
- When you are done, produce a concise Markdown summary: what you changed,
  why each change was needed, and any caveats or follow-up recommendations.
  Begin this summary with the exact line: `## Subagent Summary`
"""

_SUMMARY_REQUEST = (
    "You have finished the task. Now write your final Markdown summary. "
    "Start with `## Subagent Summary`, then list every file you modified or "
    "created (and why), every shell command you ran (and what it achieved), "
    "and any caveats or recommendations."
)


def run_subagent(
    task: str,
    *,
    auto_confirm: bool = False,
    dry_run: bool = False,
    model: str = "gemini-2.5-pro",
    working_dir: str | None = None,
    extra_context: str = "",
    pre_read: list[str] | None = None,
) -> SubagentResult:
    if working_dir:
        os.chdir(working_dir)

    def _subagent_confirm(prompt: str) -> bool:
        if auto_confirm:
            console.print(f"[dim][auto-confirm] {prompt}[/dim]")
            return True
        response = console.input(f"{prompt} — Allow? [bold dodger_blue1]\\[y/n][/bold dodger_blue1]: ").strip().lower()
        return response in ("y", "yes")

    tools.init_tools(console, _subagent_confirm, dry_run=dry_run, pre_read=pre_read)

    try:
        credentials, project_id = google_auth_default()
        if not project_id: project_id = getattr(credentials, "quota_project_id", None)
        if not project_id: project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            return SubagentResult(task=task, summary="", success=False, error="Could not detect GCP project.")
        client = genai.Client(vertexai=True, project=project_id, location="us-central1")
    except Exception as e:
        return SubagentResult(task=task, summary="", success=False, error=f"Auth failed: {e}")

    full_task = task.strip()
    if extra_context: full_task += f"\n\n## Additional context\n{extra_context.strip()}"

    system_instruction = _SUBAGENT_SYSTEM_PROMPT
    vagent_path = Path(".vagent")
    if vagent_path.exists():
        try:
            if vagent_content := vagent_path.read_text(encoding="utf-8").strip():
                system_instruction += f"\n\n## Project-specific context\n{vagent_content}"
        except Exception: pass

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[tools.local_tools],
        temperature=0.3,
    )

    history = [types.Content(role="user", parts=[types.Part(text=full_task)])]

    console.print()
    console.print(Panel(f"[bold white]{markup_escape(task)}[/bold white]", title="[bold medium_purple1]Gemini Subagent — Task[/bold medium_purple1]", border_style="dodger_blue1"))

    summary_text = ""
    error_counts: dict[str, int] = {}

    try:
        for turn in range(30):
            with console.status("[bold medium_purple1]Gemini thinking...[/bold medium_purple1]", spinner="dots"):
                try: response = client.models.generate_content(model=model, contents=history, config=config)
                except Exception as e:
                    return SubagentResult(task=task, summary=summary_text, changes=[ChangeRecord(**c) for c in tools.get_change_records()], success=False, error=f"API error on turn {turn}: {e}")

            if not response.candidates or not response.candidates[0].content:
                return SubagentResult(task=task, summary=summary_text, changes=[ChangeRecord(**c) for c in tools.get_change_records()], success=False, error="Empty response.")

            candidate = response.candidates[0]
            parts = candidate.content.parts or []
            history.append(candidate.content)

            function_calls = [p.function_call for p in parts if p.function_call and p.function_call.name]

            if not function_calls:
                text = "\n".join(p.text for p in parts if p.text)
                if "## Subagent Summary" in text:
                    summary_text = text
                    break
                history.append(types.Content(role="user", parts=[types.Part(text=_SUMMARY_REQUEST)]))
                continue

            response_parts: list[types.Part] = []
            stuck = False
            for fc in function_calls:
                fn_name = fc.name
                fn_args = dict(fc.args)

                console.print(f"[dodger_blue1]>[/dodger_blue1] [dim white]{fn_name}[/dim white]", end=" ")

                handler = tools.TOOL_DISPATCH.get(fn_name)
                result = handler(fn_args) if handler else f"ERROR: Unknown tool '{fn_name}'"

                if result.startswith("ERROR:"):
                    error_counts[fn_name] = error_counts.get(fn_name, 0) + 1
                    console.print("[bright_red]FAIL[/bright_red]")
                    console.print(Panel(markup_escape(result), title="[color(196)]Tool Error[/color(196)]", border_style="color(196)", box=box.SQUARE))
                    if error_counts.get(fn_name, 0) >= 3:
                        console.print("[bold yellow]Agent stuck in error loop. Aborting.[/bold yellow]")
                        stuck = True
                        break
                else:
                    console.print("[bright_green]OK[/bright_green]")

                response_parts.append(types.Part(function_response=types.FunctionResponse(name=fn_name, response={"content": result})))

            if stuck:
                history.pop()
                break

            history.append(types.Content(role="user", parts=response_parts))
    finally:
        tools.cleanup_background_jobs()

    if not summary_text:
        summary_text = "*(Gemini did not produce a summary — check the change log above.)*"

    return SubagentResult(
        task=task,
        summary=summary_text,
        changes=[ChangeRecord(**c) for c in tools.get_change_records()],
        success=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemini as a coding subagent supervised by Claude.", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("task", help="Natural-language task description for Gemini.")
    parser.add_argument("--auto-confirm", action="store_true", help="Automatically approve all tool calls.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate writes and shell commands.")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Gemini model ID.")
    parser.add_argument("--context", default="", help="Extra context to append to the task prompt.")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Machine-readable mode.")
    parser.add_argument("--dir", default="", help="Working directory.")
    parser.add_argument("--pre-read", action="append", default=[], help="List of files already read by the main agent.")
    args = parser.parse_args()

    if args.json_output:
        global console
        console = Console(file=sys.stderr)

    result = run_subagent(
        task=args.task, auto_confirm=args.auto_confirm, dry_run=args.dry_run,
        model=args.model, extra_context=args.context, working_dir=args.dir or None,
        pre_read=args.pre_read,
    )

    if args.json_output:
        import json
        print(json.dumps({
            "success": result.success, "error": result.error, "summary": result.summary,
            "changes": [{"kind": c.kind, "detail": c.detail, "outcome": c.outcome} for c in result.changes],
        }))
    else:
        result.display()

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
