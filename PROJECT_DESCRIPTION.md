# Vagent (Vertex Agent)

## One-Liner
A local agentic REPL that lets you converse with Google's Gemini 2.5 Pro model, which can autonomously read/write files, execute shell commands, search code, and fetch URLs on your machine to complete multi-step tasks.

## Tags
ai, cli, python, llm, vertex-ai, gcp, developer-tools, agentic, repl, terminal-ui

## Tech Stack
- **Google Vertex AI SDK (`google-genai`)** — sends chat history + tool declarations to Gemini 2.5 Pro; parses `FunctionCall` responses to dispatch local tools
- **Python 3.11+** — core agent logic structured as a Python package in `src/vagent/`
- **Rich** — full terminal UI: syntax-highlighted code blocks, color-coded unified diffs with line-number gutters, panels, tables, and markdown rendering
- **Prompt Toolkit** — multiline REPL input with custom key bindings and a continuation prompt
- **Google Auth (`google-auth`)** — GCP Application Default Credentials resolution with multi-level project ID fallback (ADC → quota project → env var)
- **Python `subprocess`** — executes shell commands (sync with 60s timeout, or async background jobs writing to temp files)
- **Python `threading`** — runs `generate_content()` in a background thread so the main thread can poll for an Escape-key cancel signal
- **Platform-specific terminal APIs** — `msvcrt` (Windows) and `select`/`tty` (Unix) for non-blocking keyboard input
- **`html.parser` (stdlib)** — custom `HTMLParser` subclass strips tags from fetched web pages before returning plain text to the model
- **`setuptools` + `pyproject.toml`** — PEP 517 build; installs as `vagent` CLI entry point

## Architecture
Vagent is a single-process, in-memory agentic loop. The user types a prompt into a `prompt_toolkit` REPL; the message is appended to a growing `chat_history` list and sent to Gemini 2.5 Pro via the Vertex AI SDK with 15 `FunctionDeclaration` tool schemas attached. If the model responds with `FunctionCall` parts, Vagent dispatches each call to its local Python implementation (file I/O, shell exec, git, grep, URL fetch), collects `FunctionResponse` parts, appends them to history, and loops — repeating until the model produces a plain text reply. Mutation tools (`write_file`, `edit_file`, `execute_bash`, `git_commit`) gate on an explicit user confirmation prompt before executing. When accumulated token count exceeds 100k, the agent calls the model again with a summarization prompt and collapses history to ~3 turns to stay within the context window.

## Technical Challenges

- **Escape-key cancellation of in-flight API calls**: `generate_content()` is blocking; to allow mid-request interruption, it runs in a daemon thread while the main thread polls a platform-specific non-blocking key check (`msvcrt.kbhit()` on Windows, `select()` in raw mode on Unix) every 50 ms. On Escape, the thread result is discarded and the REPL returns immediately.

- **Claude Code-style diff rendering**: Implemented a custom unified diff parser that tracks old/new line numbers across `@@ ... @@` hunk headers. Each diff line is rendered as a Rich `Text` object with full-row background color (green for additions, red for deletions), a right-justified line-number gutter, dimmed context lines, hunk separators, and a 80-changed-line truncation guard — all without a diff-rendering library.

- **Read-before-write enforcement**: A session-scoped `_read_files` set records every resolved absolute path passed to `read_file`. `write_file` and `edit_file` reject writes to paths not in the set, preventing the model from overwriting files it hasn't inspected. New file paths bypass the check.

- **Background job management**: `execute_bash_background()` spawns `subprocess.Popen()`, immediately returns an integer job ID, and directs stdout/stderr to temp files. `get_job_output(job_id)` polls `process.poll()` for completion, returns partial output while running, and cleans up temp files and the job registry on finish.

- **Token-threshold auto-compaction**: After each model reply, the agent reads `response.usage_metadata.total_token_count`. Above 100k tokens it calls the model with a compress-history prompt, then replaces the full `chat_history` with three turns: the summary, an acknowledgment, and the most recent user message — resuming seamlessly.

- **GCP project auto-detection**: The agent tries `google.auth.default()` credentials' `project_id`, then `quota_project_id`, then the `GOOGLE_CLOUD_PROJECT` env var, raising a human-readable error with `gcloud auth` instructions only if all three are absent.

- **Edit ambiguity detection**: `edit_file` counts occurrences of `old_string` in the file before replacing. If more than one match is found it returns an error instructing the model to supply more context, preventing silent wrong-instance substitution.

## Quantifiable Details
- Modular Python codebase under `src/vagent/`
- 15 tools exposed to the model via `FunctionDeclaration` schemas
- 5 slash commands (`/exit`, `/clear`, `/compact`, `/plan`, `/help`)
- Hard cap of 500 results for glob searches, 200 for grep matches
- Shell command timeout: 60 seconds
- URL fetch truncated to 12,000 characters (after HTML stripping)
- Diff display truncated at 80 changed lines
- Auto-compaction threshold: 100,000 tokens
- Escape-key poll interval: 50 ms
- Model temperature: 0.3 (low for deterministic code output)

## Deployment Status
Local CLI tool installed via `pip install -e .`. No cloud deployment — runs entirely on the user's machine against their own GCP project. Not published to PyPI. Entry point: `vagent` command after installation.

## Pre-Written Resume Bullets
- Engineered a 15-tool agentic REPL in Python, enabling Gemini 2.5 Pro to autonomously complete multi-step file editing, shell execution, and web research tasks with zero external orchestration framework.
- Eliminated accidental file overwrites by tracking every `read_file` call in a session-scoped set and rejecting mutation tool calls for unread paths, making write operations safe for autonomous model execution.
- Built cross-platform Escape-key cancellation for blocking Vertex AI API calls by running `generate_content()` in a background thread and polling platform-specific non-blocking input APIs (`msvcrt` on Windows, `select`/`tty` on Unix) at 50 ms intervals.
- Implemented Claude Code-style unified diff rendering from scratch using a custom hunk parser with per-line Rich `Text` objects, full-row background colors, and line-number gutters — without any diff-rendering dependency.
- Kept long agentic sessions viable by detecting token usage above 100k from response metadata and automatically compacting chat history to ~3 turns via a model-driven summarization call, preserving context across arbitrarily long workflows.
- Designed a background job system using `subprocess.Popen` with temp-file I/O and integer job IDs, allowing the model to dispatch long-running shell commands (builds, test suites) without blocking the REPL.

## Additional

### Slash Commands
| Command | Effect |
|---|---|
| `/plan` | Toggle plan mode — blocks all mutation tools, model describes changes only |
| `/compact` | Immediately summarize and compress conversation history |
| `/clear` | Wipe chat history and the read-files tracking set |
| `/exit` | Quit the agent |
| `/help` or `?` | Print tool and command reference table |

### `.vagent` File
The agent loads a `.vagent` file from the current directory at startup and appends its contents to the system prompt. This lets each project define task-specific context, coding conventions, or constraints without modifying the agent itself.

### Plan Mode
`/plan` toggles a mode that injects a constraint into the system prompt and adds `write_file`, `edit_file`, `execute_bash`, `execute_bash_background`, `git_add`, and `git_commit` to a blocked-tools set. The model can still call read-only tools (read, grep, glob, git status/diff/log) but any attempt to call a blocked tool returns an error instructing it to describe the change instead.

### Dry-Run Mode
`--dry-run` at startup simulates write and execute operations — confirmation prompts still appear, but writes are not persisted and shell commands are not executed. Useful for testing prompts against a codebase without risk.
