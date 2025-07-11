AI-Powered Code Analyzer & Q&A Assistant
This is a powerful, multi-language code analysis tool that uses Large Language Models (LLMs) to provide in-depth insights into your codebase. It goes beyond simple linting by generating comprehensive reports on functionality, potential bugs, and architecture. After the initial analysis, it drops you into an interactive Q&A session, allowing you to "talk" to your code.

This tool is built with Python and LangChain, offering the flexibility to use either Google's Gemini models or a custom, OpenAI-compatible API endpoint provided by your organization.

Features
Multi-Language Support: Analyze codebases written in Python, C#, C/C++, Java, Go, and Rust.

Flexible LLM Providers: Seamlessly switch between Google Gemini and any custom OpenAI-compatible API endpoint.

High-Speed Parallel Processing: Leverages concurrent processing to analyze multiple code files at once, significantly speeding up the analysis of large projects.

Comprehensive Code Reports: Generates a detailed report covering:
    * A high-level architectural summary.
    * A consolidated list of potential bugs, anti-patterns, and vulnerabilities.
    * A complete list of unique dependencies.

Interactive Q&A Sessions: After the analysis, ask specific questions about your code in natural language (e.g., "What does this function do?" or "Where is the database logic handled?").

Real-time Progress Tracking: A dynamic stopwatch shows the elapsed time during the analysis phase.

**Enhanced Caching:** Analysis results are cached based on file content hashes. Unchanged files are not re-analyzed, speeding up subsequent runs.

**Git Integration (Optional):** Analyze only the files that have been changed (staged, unstaged, or untracked) since the last commit by using the `--use-git` flag. This is ideal for quick analysis of recent modifications.

**Flexible Output Formats:** Reports can be generated in plain text (default), JSON, or HTML. You can also specify an output file to save the report.

Requirements
Before you begin, ensure you have Python 3.8+ installed. Then, install the required libraries using pip:

```bash
pip install langchain langchain-google-genai langchain-openai langchain-community chromadb openai tiktoken
```

Setup & Configuration
This tool requires API keys to communicate with the LLM providers. It's highly recommended to configure these using environment variables.

For Google Gemini (Default)
Obtain an API key from the Google AI Studio.
Set it as an environment variable:
```bash
# On Linux/macOS
export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"

# On Windows
set GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
```

For a Custom Endpoint
You will need your company's API endpoint URL and the corresponding API key.
Set them as environment variables:
```bash
# On Linux/macOS
export CUSTOM_API_URL="YOUR_ENDPOINT_URL"
export CUSTOM_API_KEY="YOUR_CUSTOM_API_KEY"

# On Windows
set CUSTOM_API_URL="YOUR_ENDPOINT_URL"
set CUSTOM_API_KEY="YOUR_CUSTOM_API_KEY"
```

Usage
The script is run from the command line. You can specify the codebase path, the programming language, LLM provider, and output preferences.

Basic Examples
Analyze a Python project in the current directory using Gemini:
```bash
python analyzer.py
```

Analyze a C# project in a specific directory:
```bash
python analyzer.py --language csharp --path /path/to/your/csharp/project
```

Analyze a Java project and save the report as HTML:
```bash
python analyzer.py --language java --path ./my-java-app --output-format html --output-file report.html
```

Analyze only changed Go files since the last commit and output as JSON:
```bash
python analyzer.py --language go --path ./my-go-project --use-git --output-format json
```

Analyze a C++ project using a custom LLM provider:
```bash
python analyzer.py --language cpp --llm-provider custom --path ./my_cpp_app
```

Command-Line Arguments
* `--path`: (Optional) The path to the codebase directory. Defaults to the current directory (`.`).
* `--language`: (Optional) The programming language of the codebase. Choices: `python`, `csharp`, `cpp`, `java`, `go`, `rust`. Defaults to `python`.
* `--llm-provider`: (Optional) The LLM provider to use. Choices: `gemini`, `custom`. Defaults to `gemini`.
* `--model-name`: (Optional) The model name for a custom LLM provider (e.g., `gpt-4`). Defaults to `gpt-4`.
* `--endpoint-url`: (Optional) The API endpoint URL for a custom LLM provider.
* `--api-key`: (Optional) The API key for a custom LLM provider.
* `--debug`: (Optional) Enable debug mode to print detailed file paths and Git information.
* `--use-git`: (Optional) Analyze only changed files (staged, unstaged, untracked) as reported by Git. If not a Git repo or Git is not found, falls back to a full scan.
* `--output-format`: (Optional) The output format for the report. Choices: `text`, `json`, `html`. Defaults to `text`.
* `--output-file`: (Optional) Path to save the report. If not provided, the report is printed to the console.

Generated Artifacts
The analyzer stores its generated data (cache, vector database, and report outputs) outside of the project you are analyzing. This helps keep your project directory clean and organizes analysis data centrally.

1.  **`Analysis Reports` Directory**:
    *   A folder named `Analysis Reports` is created in the same directory where the `analyzer.py` script is located.
    *   This directory will house all data related to different analysis sessions.
    *   *Requirement*: The script needs write permissions in its own directory to create `Analysis Reports` and its subdirectories.

2.  **Project-Specific Subdirectory**:
    *   Inside `Analysis Reports`, a subdirectory is created for each unique project path you analyze.
    *   The name of this subdirectory is derived from the base name of your project's path (e.g., if you analyze `/path/to/my-app`, a folder named `my-app` will be created here). Special characters in the project name are sanitized.
    *   This project-specific folder stores shared data across multiple analysis runs for that same project.

3.  **Artifacts within Project-Specific Directory**:
    *   `.analyzer_cache.json`: This file, located directly inside the project-specific folder (e.g., `Analysis Reports/my-app/.analyzer_cache.json`), stores cached analysis results for individual files of that project. This cache is reused and updated across multiple analysis runs of the *same project*, speeding up subsequent analyses.
    *   `code_db/`: This directory, also inside the project-specific folder (e.g., `Analysis Reports/my-app/code_db/`), contains the vector database for the project. This database is also reused and updated, allowing the Q&A session to draw upon an ever-improving understanding of the codebase across runs.
    *   **Report Output Files**: If you use the `--output-file <filename.ext>` argument, the generated report (e.g., HTML, JSON) is saved *inside this project-specific folder*. To prevent overwriting reports from different runs, the filename is prefixed with a timestamp: `YYYYMMDD-HHMMSS_<filename.ext>`. For example, `Analysis Reports/my-app/20231027-153000_report.html`.

**Example Structure:**
```
/path/to/Code_Analysis_Folder/
├── analyzer.py
├── Analysis Reports/
│   ├── my-app/  <----------------------- For project '/path/to/my-app'
│   │   ├── .analyzer_cache.json  <------ Shared cache for my-app
│   │   ├── code_db/  <------------------ Shared Q&A DB for my-app
│   │   │   └── ... (ChromaDB files)
│   │   ├── 20231027-153000_report.html <- Report from one run
│   │   └── 20231027-160000_analysis.json <- Report from another run
│   └── another-project/ <--------------- For project '/path/to/another-project'
│       ├── .analyzer_cache.json
│       └── code_db/
└── ... (other files)
```

You can safely delete any project-specific subdirectory (e.g., `Analysis Reports/my-app/`) if you want to clear all cached data and Q&A history for that project.

How It Works
The tool operates in three main stages:

1.  **File Discovery & Filtering:**
    *   If `--use-git` is active, it attempts to identify changed files (staged, unstaged, untracked) in the repository.
    *   Otherwise, it performs a full scan of the specified path for files matching the selected language.
    *   Files are then filtered by the chosen programming language.
2.  **Map-Reduce Analysis with Caching:**
    *   For each file to be analyzed, its content hash is calculated.
    *   If a result for this hash exists in the cache (`.analyzer_cache.json`), it's used directly.
    *   For new or modified files, the code is split into manageable chunks. These chunks are sent to the selected LLM for analysis in parallel.
    *   The individual analyses (new and cached) are then synthesized by the LLM into a single, comprehensive report.
    *   New analysis results are saved to the cache.
3.  **Vector Store Creation:** The code chunks (from all processed files, including cached ones if they were loaded) are converted into numerical representations (embeddings) and stored in a local vector database (using ChromaDB). This allows for efficient, semantic-based searching.
4.  **Interactive Q&A:** With the vector store in place, you can ask questions. The tool finds the most relevant code chunks related to your query, provides them to the LLM as context, and generates a targeted answer.

Roadmap & Future Improvements
The following items from the original roadmap have been **implemented**:
*   [x] Add support for more programming languages (e.g., Java, Go, Rust).
*   [x] Allow for different output formats for the report (e.g., JSON, HTML).
*   [x] Implement a mechanism to cache results to avoid re-analyzing unchanged files. (This was already partially present and has been verified/enhanced by ensuring all loaded docs contribute to Q&A).
*   [x] Integrate with Git to only analyze changed files between commits.

Potential future enhancements could include:
*   More sophisticated HTML reports with syntax highlighting and navigation.
*   Configuration file for presets (e.g., default language, LLM provider).
*   Support for analyzing specific functions or classes directly.
*   Deeper static analysis to complement LLM insights.
