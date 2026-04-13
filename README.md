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

- **Local Tool Execution:** The agent can execute local tools on your machine, such as reading and writing files, executing shell commands, and listing directory contents. The available tools are:
    - `read_file`: Read the full text contents of a file on the local filesystem.
    - `write_file`: Write (overwrite) a file on the local filesystem with the given content.
    - `execute_bash`: Execute a shell command on the local machine and return its stdout/stderr.
    - `list_directory`: List the files and folders inside a directory on the local filesystem.
- **Confirmation Prompt:** For security, the agent will prompt for confirmation before executing potentially dangerous commands like `write_file` and `execute_bash`.
- **Dry-Run Mode:** You can run the agent in dry-run mode to see what commands it would execute without actually making any changes to your system.
- **History Compaction:** The agent can automatically compact the conversation history to stay within the model's token limit.
- **Rich Display:** The agent uses the `rich` library to provide a rich and user-friendly command-line interface, including syntax highlighting for code blocks.
- **Slash Commands:** The agent supports slash commands for performing common tasks, such as clearing the history, exiting the agent, and displaying help.

## Dependencies

- `google-genai>=1.0.0`
- `google-auth>=2.0.0`
- `rich>=13.0.0`
- `prompt_toolkit>=3.0.0`

## Slash Commands

The following slash commands are available:

| Command     | Description                                |
| ----------- | ------------------------------------------ |
| `/exit`     | Quit the agent                             |
| `/clear`    | Wipe the conversation history              |
| `/compact`  | Summarise and compress history             |
| `/help` or `?` | Show this help                             |
