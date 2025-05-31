# SWE-Fixer: AI Assistant for Code Analysis and Correction

## 1. 📜 Project Overview
SWE-Fixer is an AI assistant that automatically analyzes and corrects errors in code files (Java and C++) when commits are made to a Git repository. It uses a local language model (Armandotrsg/qwen-cybersecurity-2.5-7b-gguf) to detect issues such as unused variables, syntax errors, unreachable code, etc., and applies the suggested corrections.

## 2. 📋 Prerequisites
- Python 3.8 or higher
- Git
- Ollama (to run the local model)
- Python virtual environment (venv)
- Download model Armandotrsg/qwen-cybersecurity-2.5-7b-gguf from hugging face

## 3. 📁 Project Structure
- `api_server.py`: Flask server that exposes the `/chat` endpoint to interact with the local model.
- `git_hook.py`: Git hook that activates on commits, analyzes modified files, and applies corrections.
- `generate_readme.py`: Script to automatically generate the README.md based on project analysis.
- `test/`: Test folder with example Python and Java files to validate the system's functionality.

## 4. 💡 How It Works
The project is based on an automated workflow:
1. When a commit is made, the Git hook (`git_hook.py`) detects the modified files.
2. The hook sends the file contents to the Flask server (`api_server.py`).
3. The server uses the local model (Armandotrsg/qwen-cybersecurity-2.5-7b-gguf) to analyze the code and generate correction suggestions.
4. The hook applies the suggested corrections and makes a new commit with the changes.
5. Hugging face link for the trained model: https://huggingface.co/Armandotrsg/qwen-cybersecurity-2.5-7b-gguf/tree/main

## 5. ⚙️ Setup and Installation
1. **Obtaining the code:**
   Ensure you have all project files on your local machine.

2. **Creating and activating the virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   # or
   .venv\Scripts\activate  # Windows
   ```

3. **Installing dependencies:**
   If a `requirements.txt` file exists, run:
   ```bash
   pip install -r requirements.txt
   ```
   If not, install dependencies manually:
   ```bash
   pip install flask requests
   pip freeze > requirements.txt
   ```

## 6. ▶️ Running the Project
1. **Starting the Flask server:**
   ```bash
   cd GitFixer
   ```
   ```bash
   source venv/bin/activate
   ```
   ```bash
   python api_server.py serve
   ```
   To restart the server (if needed):
   ```bash
   cd .. && pkill -f "python.*api_server.py" && source venv/bin/activate && python api_server.py serve
   ```
   **Note:** Make sure you have port 7774 free, since this proyect runs in local host using this port.
   
   Or, from the project root:
   ```bash
   cd /GitFixer && pkill -f "python.*api_server.py" && source venv/bin/activate && python api_server.py serve
   ```

3. **Configuring the Git hook:**
   Ensure `git_hook.py` is set up as a post-commit hook in your repository.

4. **Making commits:**
   When you make commits, the hook will automatically analyze modified files and apply suggested corrections. For example:
   ```bash
   python githubTesting/GithubFixerTest/run_code_fixer_hook.py <Path to your cloned repository>
   ```
   Note: For making commits you have to your own repository and add file **run_code_fixer_hook.py** inside your folder of the github repository that you cloned. 

## 7. ✨ Usage Example
To test the system, you can create a file with syntax errors, for example:
```python
# hello.py
def main():
    x = 10
    y = 0
    z = x / y  # Error: division by zero
    print(z)

if __name__ == "__main__":
    main()
```
When you commit, the hook will detect the error and suggest a correction:
```python
# hello.py
def main():
    x = 10
    y = 1  # Corrected: avoid division by zero
    z = x / y
    print(z)

if __name__ == "__main__":
    main()
```

---

