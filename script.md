# SWE-Fixer and SWE-Agent Integration Script

## Overview
This script outlines the integration between SWE-Fixer and SWE-Agent, explaining how they work together to analyze and fix issues in code files when running GitHub commands.

## How It Works

### SWE-Fixer
- **Purpose**: SWE-Fixer is an AI assistant that automatically analyzes and corrects errors in code files (Python and Java) when commits are made to a Git repository.
- **Functionality**:
  - Detects issues such as unused variables, syntax errors, and unreachable code.
  - Uses a local language model (qwen-cybersecurity-2.5-7b-gguf) to generate correction suggestions.
  - Applies the suggested corrections and makes a new commit with the changes.

### SWE-Agent
- **Purpose**: SWE-Agent is responsible for understanding bug reports and generating precise and contextually relevant code patches.
- **Functionality**:
  - Utilizes Large Language Models (LLMs) to interpret bug reports.
  - Works in conjunction with SWE-Fixer to automate the patch generation process.
  - Aims to accelerate bug fixing and improve software maintenance workflows.

## Integration Process
1. **Commit Trigger**: When a commit is made, the Git hook (`git_hook.py`) activates.
2. **File Analysis**: The hook sends the modified file contents to the Flask server (`api_server.py`).
3. **Model Interaction**: The server uses the local model to analyze the code and generate correction suggestions.
4. **Patch Application**: SWE-Agent interprets the suggestions and applies the necessary patches to the code.
5. **Commit Changes**: The corrected code is committed back to the repository.

## Algorithm Used
The project utilizes the BM5 algorithm, which is a variant of the Boyer-Moore string search algorithm. This algorithm is particularly efficient for searching patterns in text, making it suitable for identifying specific code issues and generating relevant patches.

## Project Structure
- `api_server.py`: Flask server that exposes the `/chat` endpoint to interact with the local model.
- `git_hook.py`: Git hook that activates on commits, analyzes modified files, and applies corrections.
- `generate_readme.py`: Script to automatically generate the README.md based on project analysis.
- `test/`: Test folder with example Python and Java files to validate the system's functionality.

## Example Command
To trigger the process, run:
```bash
git add <file> && git commit -m "Test: <file> with syntax errors"
```

This command will activate the SWE-Fixer and SWE-Agent integration, leading to automatic analysis and correction of the specified file.

---

This script is part of the SWE-Fixer project, designed to enhance code quality and streamline the bug-fixing process.