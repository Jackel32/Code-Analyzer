AI-Powered Code Analyzer & Q&A Assistant
This is a powerful, multi-language code analysis tool that uses Large Language Models (LLMs) to provide in-depth insights into your codebase. It goes beyond simple linting by generating comprehensive reports on functionality, potential bugs, and architecture. After the initial analysis, it drops you into an interactive Q&A session, allowing you to "talk" to your code.

This tool is built with Python and LangChain, offering the flexibility to use either Google's Gemini models or a custom, OpenAI-compatible API endpoint provided by your organization.

Features
Multi-Language Support: Analyze codebases written in Python, C#, and C/C++.

Flexible LLM Providers: Seamlessly switch between Google Gemini and any custom OpenAI-compatible API endpoint.

High-Speed Parallel Processing: Leverages concurrent processing to analyze multiple code files at once, significantly speeding up the analysis of large projects.

Comprehensive Code Reports: Generates a detailed report covering:

A high-level architectural summary.

A consolidated list of potential bugs, anti-patterns, and vulnerabilities.

A complete list of unique dependencies.

Interactive Q&A Sessions: After the analysis, ask specific questions about your code in natural language (e.g., "What does this function do?" or "Where is the database logic handled?").

Real-time Progress Tracking: A dynamic stopwatch shows the elapsed time during the analysis phase.

Requirements
Before you begin, ensure you have Python 3.8+ installed. Then, install the required libraries using pip:

pip install langchain langchain-google-genai langchain-openai langchain-community chromadb openai tiktoken

Setup & Configuration
This tool requires API keys to communicate with the LLM providers. It's highly recommended to configure these using environment variables.

For Google Gemini (Default)
Obtain an API key from the Google AI Studio.

Set it as an environment variable:

# On Linux/macOS
export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"

# On Windows
set GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"

For a Custom Endpoint
You will need your company's API endpoint URL and the corresponding API key.

Set them as environment variables:

# On Linux/macOS
export CUSTOM_API_URL="YOUR_ENDPOINT_URL"
export CUSTOM_API_KEY="YOUR_CUSTOM_API_KEY"

# On Windows
set CUSTOM_API_URL="YOUR_ENDPOINT_URL"
set CUSTOM_API_KEY="YOUR_CUSTOM_API_KEY"

Usage
The script is run from the command line. You can specify the codebase path, the programming language, and the LLM provider.

Basic Examples
Analyze a Python project in the current directory using Gemini:

python analyzer.py

Analyze a C# project in a specific directory:

python analyzer.py --language csharp --path /path/to/your/csharp/project

Analyze a C++ project using a custom LLM provider:

python analyzer.py --language cpp --llm-provider custom --path ./my_cpp_app

![image](https://github.com/user-attachments/assets/4897f5cb-30a3-4b3f-be5f-8d5cb76d6d97)

How It Works
The tool operates in three main stages:

Map-Reduce Analysis: The script first scans your project for code files of the specified language. It breaks the code into manageable chunks and sends them to the LLM for analysis in parallel. The individual analyses are then synthesized into a single, comprehensive report.

Vector Store Creation: The code chunks are converted into numerical representations (embeddings) and stored in a local vector database (using ChromaDB). This allows for efficient, semantic-based searching.

Interactive Q&A: With the vector store in place, you can ask questions. The tool finds the most relevant code chunks related to your query, provides them to the LLM as context, and generates a targeted answer.

Roadmap & Future Improvements
[ ] Add support for more programming languages (e.g., Java, Go, Rust).

[ ] Allow for different output formats for the report (e.g., JSON, HTML).

[ ] Implement a mechanism to cache results to avoid re-analyzing unchanged files.

[ ] Integrate with Git to only analyze changed files between commits.
