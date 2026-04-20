# Vertex Agent

## Description

This project provides an agentic REPL (Read-Eval-Print Loop) for interacting with Vertex AI's Gemini models. It allows you to have a conversation with the model, and the model can execute local tools on your machine to perform tasks.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/vagent.git
   cd vagent
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```

## Usage

To start the agent, run the following command:

```bash
vagent
```

You can also run the agent in dry-run mode, which will simulate the execution of commands without actually performing them:

```bash
vagent --dry-run
```

## Features

- **Multimodal Vision Support:** Drag and drop images (PNG, JPG, WEBP, HEIC) right into the terminal. The agent will read the pixels and write code based on UI mockups or screenshots of bugs!
- **Native Google Search Grounding:** The agent can autonomously query Google Search in real-time to find up-to-date documentation, read StackOverflow, and provide clickable citations.
- **Sub-Agent Workflows:** Use `delegate_task` to spawn parallel Gemini instances with fresh context windows to complete massive, multi-step refactors without losing track of the main conversation.
- **Local Tool Execution:** The agent can execute local tools on your machine, such as reading and writing files, executing shell commands, and listing directory contents.
- **Long-term Memory:** The agent learns your project rules over time, saving them to a `.vagent` file so it remembers how to run your tests across sessions.
- **Confirmation Prompt:** For security, the agent will prompt for confirmation before executing potentially dangerous commands like `write_file` and `execute_bash`.
- **Dry-Run Mode:** You can run the agent in dry-run mode to see what commands it would execute without actually making any changes to your system.
- **Plan Mode:** You can run the agent in plan mode to have the model describe the changes it would make without executing them.
- **History Compaction:** The agent can automatically compact the conversation history to stay within the model's token limit.

## Dependencies

- `google-genai>=1.0.0`
- `google-auth>=2.0.0`
- `rich>=13.0.0`
- `prompt_toolkit>=3.0.0`

## Available Tools

The agent has access to the following local tools:

| Tool                      | Description                                                                                                                               |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `read_file`               | Read the full text contents of a file on the local filesystem.                                                                            |
| `write_file`              | Write (overwrite) a file on the local filesystem with the given content. (Prompts for confirmation)                                       |
| `edit_file`               | Surgically edit an existing file by replacing an exact string with new text. Token-efficient and safe. (Prompts for confirmation)         |
| `execute_bash`            | Execute a shell command on the local machine and return its stdout/stderr. (Prompts for confirmation)                                     |
| `execute_bash_background` | Start a long-running background job without blocking (e.g., servers, test watchers).                                                      |
| `get_job_output`          | Check the status, stdout, and stderr of a background job.                                                                                 |
| `ask_user`                | Pause the autonomous loop to ask the user a specific question, waiting for their response before continuing.                              |
| `save_memory`             | Remember a fact or project rule permanently across sessions by saving it to `.vagent`.                                                    |
| `delegate_task`           | Spawn a secondary Gemini subagent to autonomously complete a multi-step task in parallel, returning a summary.                            |
| `list_directory`          | List the files and folders inside a directory on the local filesystem.                                                                    |
| `glob_files`              | Find files across the project matching a glob pattern (e.g. `**/*.py`).                                                                   |
| `grep_files`              | Regex search across file contents in the project to find specific code or variables.                                                      |
| `fetch_url`               | Fetch plain text content from a URL (useful for reading docs).                                                                            |
| `git_*`                   | Suite of Git tools (`git_status`, `git_diff`, `git_log`, `git_add`, `git_commit`) to autonomously track and commit changes.               |

## Slash Commands

The following slash commands are available:

| Command     | Description                                |
| ----------- | ------------------------------------------ |
| `/exit`     | Quit the agent                             |
| `/clear`    | Wipe the conversation history              |
| `/compact`  | Summarise and compress history             |
| `/plan`     | Toggle plan mode                           |
| `/help` or `?` | Show this help                             |

