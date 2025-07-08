import os
import sys
import time
import json
import hashlib # Import for file hashing
import subprocess # Import for Git integration
import threading
import argparse
import concurrent.futures
# json import already present from config changes
# os import already present from config changes
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader, TextLoader # DirectoryLoader might not be used now
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# OutputParserException might not be used now
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

from static_analyzers import run_pylint, run_cppcheck # Import static analyzer functions

# --- Caching Functions ---
CACHE_FILENAME = ".analyzer_cache.json"

def load_cache(code_path: str):
    """Loads the analysis cache from a JSON file in the specified code_path."""
    cache_file_path = os.path.join(code_path, CACHE_FILENAME)
    # Check args.debug if available, otherwise default to False or handle appropriately
    debug_mode = hasattr(args, 'debug') and args.debug
    if not os.path.exists(cache_file_path):
        if debug_mode:
            print(f"  [DEBUG Cache] Cache file not found at {cache_file_path}, creating new cache.")
        return {}
    if debug_mode:
        print(f"  [DEBUG Cache] Attempting to load cache from {cache_file_path}")
    with open(cache_file_path, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            if debug_mode:
                print(f"  [DEBUG Cache] JSONDecodeError reading cache from {cache_file_path}. Returning empty cache.")
            return {}

def save_cache(cache, code_path: str):
    """Saves the analysis cache to a JSON file in the specified code_path."""
    cache_file_path = os.path.join(code_path, CACHE_FILENAME)
    debug_mode = hasattr(args, 'debug') and args.debug
    if debug_mode:
        print(f"  [DEBUG Cache] Saving cache to {cache_file_path}")
    with open(cache_file_path, 'w') as f:
        json.dump(cache, f, indent=2)

def get_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# --- Git Integration ---
def get_changed_files_git(repo_path: str, debug: bool = False):
    try:
        subprocess.run(["git", "-C", repo_path, "rev-parse"], check=True, capture_output=True, timeout=5)
        staged_result = subprocess.run(
            ["git", "-C", repo_path, "diff", "--name-only", "--cached"],
            capture_output=True, text=True, check=True, timeout=5
        )
        unstaged_result = subprocess.run(
            ["git", "-C", repo_path, "diff", "--name-only"],
            capture_output=True, text=True, check=True, timeout=5
        )
        untracked_result = subprocess.run(
            ["git", "-C", repo_path, "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, check=True, timeout=5
        )
        changed_files = set()
        if staged_result.stdout: changed_files.update(staged_result.stdout.strip().split('\n'))
        if unstaged_result.stdout: changed_files.update(unstaged_result.stdout.strip().split('\n'))
        if untracked_result.stdout: changed_files.update(untracked_result.stdout.strip().split('\n'))
        changed_files = {f for f in changed_files if f}
        if debug:
            print(f"  [DEBUG Git] Staged: {staged_result.stdout.strip().splitlines() if staged_result.stdout else 'None'}")
            print(f"  [DEBUG Git] Unstaged: {unstaged_result.stdout.strip().splitlines() if unstaged_result.stdout else 'None'}")
            print(f"  [DEBUG Git] Untracked: {untracked_result.stdout.strip().splitlines() if untracked_result.stdout else 'None'}")
            print(f"  [DEBUG Git] Combined changed: {list(changed_files)}")
        return list(changed_files)
    except Exception as e: # More general exception handling
        if debug: print(f"  [DEBUG Git] Error in get_changed_files_git: {e}")
        return None


# --- Main Analysis Script ---
def generate_full_report(code_path: str, artifact_path: str, llm, language: str, config: dict, debug: bool = False):
    """
    Loads a codebase from code_path, performs static analysis, analyzes with LLM using a
    map-reduce strategy with caching, and returns results.
    Returns: tuple (llm_summary_report_text, all_document_objects_for_vectorstore, all_static_analysis_findings, llm_cache)
    The llm_cache is returned to allow HTML/Text report generation to access per-file LLM analyses.
    """
    print("--- Part 1: Starting Full Report Generation ---")

    LANGUAGE_MAP = { # For LangChain text splitting
        "python": Language.PYTHON, "csharp": Language.CSHARP, "cpp": Language.CPP,
        "java": Language.JAVA, "go": Language.GO, "rust": Language.RUST
    }
    GLOB_MAP = { # For file discovery
        "python": ["*.py"], "csharp": ["*.cs"],
        "cpp": ["*.c", "*.cpp", "*.h", "*.hpp", "*.cc", "*.cxx", "*.hh", "*.hxx"],
        "java": ["*.java"], "go": ["*.go"], "rust": ["*.rs"],
    }

    if language not in config.get("supported_languages", []): # Check against config
        print(f"Error: Language '{language}' is not in supported_languages configuration.")
        if language not in GLOB_MAP: # Further check if we can even find files
            print(f"Error: Unsupported language '{language}' for file scanning.")
            sys.exit(1)

    expected_extensions = [pattern.replace('*.', '.') for pattern in GLOB_MAP.get(language, [])]
    if not expected_extensions:
        print(f"Warning: No glob patterns defined for language '{language}'. Cannot find files.")
        return None, [], {}, {} # llm_summary, docs, static_findings, llm_cache_for_report

    if debug: print(f"  [DEBUG] Expected extensions for {language}: {expected_extensions}")

    print(f"\n[Step 1] Scanning for {language} files in '{code_path}'...")
    files_to_process = []
    # Access args through config or pass it if needed for use_git, or rely on config's value
    use_git_flag = config.get("use_git_by_default", False)
    if hasattr(args, 'use_git') and args.use_git: # If CLI flag is set, it overrides
        use_git_flag = args.use_git

    if use_git_flag:
        print("  -> Attempting to use Git to find changed files.")
        git_files = get_changed_files_git(code_path, debug)
        if git_files is not None:
            print(f"  -> Found {len(git_files)} changed files via Git (pre-filter).")
            files_to_process = [
                os.path.join(code_path, f)
                for f in git_files
                if os.path.splitext(f)[1] in expected_extensions and os.path.isfile(os.path.join(code_path, f))
            ]
            if files_to_process: print(f"  -> Analyzing {len(files_to_process)} {language} file(s) changed via Git.")
            else: print(f"  -> No changed {language} files via Git or error. Falling back to full scan.")
        else: print("  -> Git not available/repo. Falling back to full scan.")

    if not files_to_process: # Fallback or default scan
        if not use_git_flag: print("  -> Performing full scan.")
        from glob import glob
        files_to_process_set = set()
        for dirpath, _, _ in os.walk(code_path):
            for lang_pattern in GLOB_MAP.get(language, []):
                for f_path in glob(os.path.join(dirpath, lang_pattern)):
                    if os.path.isfile(f_path): files_to_process_set.add(os.path.abspath(f_path))
        files_to_process = [f for f in list(files_to_process_set) if os.path.splitext(f)[1] in expected_extensions]

    if debug: print(f"\n--- Files to Process (FINAL DEBUG) ---\n  -> " + "\n  -> ".join(files_to_process) + "\n----------------------------------\n")
    if not files_to_process:
        print(f"Warning: No {language} files found in '{code_path}'.")
        return None, [], {}, {}

    print(f"Found {len(files_to_process)} file(s) to analyze for {language}.")

    # Load LLM cache
    llm_cache = load_cache(artifact_path)
    files_requiring_live_llm_analysis = []
    cached_llm_results_for_files = [] # Stores full analysis strings for cached files
    all_document_objects_for_vectorstore = []
    all_static_analysis_findings = {}

    print("\n[Step 1.5] Performing Static Analysis and Preparing for LLM...")
    for file_path in files_to_process:
        current_file_static_results = {}
        if language == "python" and config.get("pylint_enabled", True) and file_path.endswith(".py"):
            if debug: print(f"  -> Running Pylint on {os.path.basename(file_path)}...")
            pylint_findings = run_pylint(file_path, config_path=config.get("pylint_rcfile"))
            if pylint_findings: current_file_static_results["pylint"] = pylint_findings
        elif language == "cpp" and config.get("cppcheck_enabled", True) and any(file_path.endswith(ext) for ext in GLOB_MAP.get("cpp",[])):
            if debug: print(f"  -> Running Cppcheck on {os.path.basename(file_path)}...")
            cppcheck_findings = run_cppcheck(file_path)
            if cppcheck_findings: current_file_static_results["cppcheck"] = cppcheck_findings

        if current_file_static_results:
            all_static_analysis_findings[file_path] = current_file_static_results
            if debug: print(f"  -> Static analysis for {os.path.basename(file_path)}: Results captured.")

        file_hash = get_file_hash(file_path)
        if file_hash in llm_cache:
            if debug: print(f"  -> [CACHE LLM] Found for {os.path.basename(file_path)}")
            cached_llm_results_for_files.append(llm_cache[file_hash])
        else:
            if debug: print(f"  -> [LIVE LLM] Queuing for {os.path.basename(file_path)}")
            files_requiring_live_llm_analysis.append((file_path, file_hash))

        try:
            docs = TextLoader(file_path).load()
            if docs: all_document_objects_for_vectorstore.extend(docs)
        except Exception as e: print(f"Warning: Could not load {file_path} for vector store: {e}")

    newly_cached_llm_results = {} # Stores {hash: combined_analysis_string} for new results
    if files_requiring_live_llm_analysis:
        live_llm_docs_to_split = [TextLoader(fp).load()[0] for fp, _ in files_requiring_live_llm_analysis]
        print(f"\n[Step 2] Splitting {len(live_llm_docs_to_split)} file(s) for LLM...")

        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=LANGUAGE_MAP.get(language, Language.PYTHON), # Fallback to Python splitter for generic
            chunk_size=1000, chunk_overlap=150
        ) if language in LANGUAGE_MAP else RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        if language not in LANGUAGE_MAP and debug: print(f"  [DEBUG] Using generic splitter for {language}")

        chunks_for_llm_processing = []
        for i, doc_to_split in enumerate(live_llm_docs_to_split):
            file_path, file_hash = files_requiring_live_llm_analysis[i]
            static_findings_str = json.dumps(all_static_analysis_findings.get(file_path, {}))
            if len(static_findings_str) > 1500: static_findings_str = static_findings_str[:1500] + "...(truncated)"

            doc_chunks = text_splitter.split_documents([doc_to_split])
            for chunk_content in doc_chunks:
                chunks_for_llm_processing.append({
                    "text_content": chunk_content.page_content,
                    "file_hash_for_caching": file_hash,
                    "static_analysis_context": static_findings_str
                })

        total_chunks = len(chunks_for_llm_processing)
        print(f"Split into {total_chunks} chunks for LLM analysis.")
        map_template = (
            f"Analyze the following chunk of {language} code. Consider these static analysis findings for the entire file: {{static_analysis_context}}. "
            "Focus on the code chunk. Code Chunk: ```\n{{text_content}}\n```\nYour Analysis:"
        )
        map_prompt = PromptTemplate.from_template(map_template)
        llm_output_parser = StrOutputParser()
        map_chain = map_prompt | llm | llm_output_parser

        print(f"\n[Step 3] Running live LLM analysis on {total_chunks} chunks...")
        start_time = time.time()
        temp_chunk_results_by_hash = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_meta = {
                executor.submit(map_chain.invoke, {"text_content": c["text_content"], "static_analysis_context": c["static_analysis_context"]}): c
                for c in chunks_for_llm_processing
            }
            for i, future in enumerate(concurrent.futures.as_completed(future_to_meta)):
                meta = future_to_meta[future]
                f_hash = meta["file_hash_for_caching"]
                try:
                    res = future.result()
                    if f_hash not in temp_chunk_results_by_hash: temp_chunk_results_by_hash[f_hash] = []
                    temp_chunk_results_by_hash[f_hash].append(res)
                    elapsed = time.time() - start_time
                    print(f"  -> LLM Mapping... [{int(elapsed//60):02d}:{int(elapsed%60):02d}] Progress: {i+1}/{total_chunks}\r", end="")
                except Exception as e: print(f"\nError processing chunk for hash {f_hash}: {e}")
        print("\n  -> Live LLM analysis of chunks complete.")

        for f_hash, analyses_list in temp_chunk_results_by_hash.items():
            newly_cached_llm_results[f_hash] = "\n\n---\nChunk Analysis:\n---\n\n".join(analyses_list)

        llm_cache.update(newly_cached_llm_results)
        save_cache(llm_cache, artifact_path)
        if debug: print(f"  [DEBUG Cache] Saved {len(newly_cached_llm_results)} new LLM analyses.")

    print("\n[Step 4] Reducing all LLM analyses into the final report...")
    all_llm_analyses_for_reduce = cached_llm_results_for_files + list(newly_cached_llm_results.values())

    llm_summary_report_text = "No LLM analysis available to generate a summary."
    if all_llm_analyses_for_reduce:
        doc_summaries_str = "\n\n=====\nNext File Analysis:\n=====\n\n".join(all_llm_analyses_for_reduce)
        reduce_template = (
            "Synthesize these code analyses into a single high-level summary report. Identify common themes and critical issues. "
            "File Analyses:\n{doc_summaries}\n\nYour High-Level Summary:"
        )
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        reduce_chain = reduce_prompt | llm | StrOutputParser()
        llm_summary_report_text = reduce_chain.invoke({"doc_summaries": doc_summaries_str})
    else:
        print("Warning: No LLM analyses (cached or live) to reduce.")

    # Return all necessary data for report generation outside this function
    return llm_summary_report_text, all_document_objects_for_vectorstore, all_static_analysis_findings, llm_cache, files_to_process


def create_vector_store(chunks, embeddings, code_path: str, debug: bool = False): # Added debug
    print("\n--- Part 2: Creating Vector Store for Q&A ---")
    if not chunks:
        print("No code chunks available to create a vector store.")
        return None
    persist_directory = os.path.join(code_path, "code_db")
    if debug: print(f"  [DEBUG Vector Store] Persisting to: {persist_directory}")
    try:
        print(f"[Step 1] Creating/loading vector store at {persist_directory}...")
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
        print(f"Vector store created/updated and saved to '{persist_directory}'.")
        return vector_store
    except Exception as e:
        print(f"An error occurred during vector store creation: {e}")
        return None

def start_qa_session(llm, vector_store):
    print("\n--- Part 3: Interactive Code Q&A ---")
    if not vector_store:
        print("Vector store not available. Cannot start Q&A session.")
        return
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
    print("\nCode Q&A session started. Type 'exit' or 'quit' to end.")
    while True:
        query = input("\nAsk a question: ")
        if query.lower() in ['exit', 'quit']:
            print("Exiting Q&A session.")
            break
        try:
            answer = qa_chain.invoke(query)
            print("\nAnswer:", answer['result'])
        except Exception as e: print(f"An error occurred during Q&A: {e}")

# --- Configuration Loading ---
DEFAULT_CONFIG = {
    "default_language": "python", "llm_provider": "gemini",
    "model_name": "gemini-1.5-flash-latest", "custom_llm_endpoint_url": None,
    "custom_llm_api_key_env_var": "CUSTOM_API_KEY",
    "supported_languages": ["python", "csharp", "cpp", "java", "go", "rust"],
    "output_format": "text", "use_git_by_default": False, "debug_mode_by_default": False,
    "pylint_enabled": True, "pylint_rcfile": None, # Path to .pylintrc
    "cppcheck_enabled": True # No specific config file for cppcheck in basic wrapper yet
}

def load_config():
    config = DEFAULT_CONFIG.copy()
    paths_to_check = [
        os.path.expanduser("~/.config/analyzer/config.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"),
        os.path.join(os.getcwd(), "config.json")
    ]
    loaded_path = None
    for path in paths_to_check:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f: config.update(json.load(f))
                loaded_path = path
                break
            except Exception as e: print(f"Warning: Error loading config {path}: {e}")
    debug_mode = hasattr(args, 'debug') and args.debug # Check if args is defined
    if not debug_mode and "debug_mode_by_default" in config: # Fallback to config if args not parsed yet
        debug_mode = config["debug_mode_by_default"]
    if debug_mode:
        if loaded_path: print(f"  [Config] Loaded configuration from: {loaded_path}")
        else: print("  [Config] No configuration file found. Using default settings.")
    return config

def generate_default_config_file(path_to_save=os.path.expanduser("~/.config/analyzer/config.json")):
    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
    with open(path_to_save, 'w') as f: json.dump(DEFAULT_CONFIG, f, indent=2)
    print(f"Generated default configuration file at: {path_to_save}")

# --- Main Execution Block ---
if __name__ == "__main__":
    config = load_config()
    parser = argparse.ArgumentParser(description="Analyze a codebase with LangChain.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--path", type=str, default=".", help="Path to codebase.")
    parser.add_argument("--language", type=str, default=config.get("default_language"), choices=config.get("supported_languages"), help="Programming language.")
    parser.add_argument("--llm-provider", type=str, default=config.get("llm_provider"), choices=["gemini", "custom"], help="LLM provider.")
    parser.add_argument("--model-name", type=str, default=config.get("model_name"), help="LLM model name.")
    parser.add_argument("--endpoint-url", type=str, default=config.get("custom_llm_endpoint_url"), help="Custom LLM endpoint URL.")
    parser.add_argument("--api-key", type=str, default=None, help=f"Custom LLM API key (else uses env var {config.get('custom_llm_api_key_env_var')}).")
    parser.add_argument("--debug", action="store_true", default=config.get("debug_mode_by_default"), help="Enable debug prints.")
    parser.add_argument("--use-git", action="store_true", default=config.get("use_git_by_default"), help="Analyze Git changed files.")
    parser.add_argument("--output-format", type=str, default=config.get("output_format"), choices=["text", "json", "html"], help="Report output format.")
    parser.add_argument("--output-file", type=str, default=None, help="Save report to file (optional).")
    parser.add_argument("--create-config", action="store_true", help="Create default config and exit.")
    # Add CLI args for static analysis tool toggles, overriding config
    parser.add_argument("--pylint-enabled", dest='pylint_enabled_cli', action=argparse.BooleanOptionalAction, help="Enable/Disable Pylint (overrides config).")
    parser.add_argument("--cppcheck-enabled", dest='cppcheck_enabled_cli', action=argparse.BooleanOptionalAction, help="Enable/Disable Cppcheck (overrides config).")


    args = parser.parse_args()

    if args.create_config:
        generate_default_config_file()
        sys.exit(0)

    # Update config with CLI overrides for static analysis tools
    if args.pylint_enabled_cli is not None: config["pylint_enabled"] = args.pylint_enabled_cli
    if args.cppcheck_enabled_cli is not None: config["cppcheck_enabled"] = args.cppcheck_enabled_cli


    effective_api_key = args.api_key
    if not effective_api_key and args.llm_provider == "custom":
        effective_api_key = os.getenv(config.get("custom_llm_api_key_env_var"))
        if not effective_api_key and args.debug: print(f"Warning: Custom LLM API key env var {config.get('custom_llm_api_key_env_var')} not set.")

    effective_endpoint_url = args.endpoint_url
    if not effective_endpoint_url and args.llm_provider == "custom":
        effective_endpoint_url = config.get("custom_llm_endpoint_url")
        if not effective_endpoint_url and args.debug: print("Warning: Custom LLM Endpoint URL not set in CLI or config.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    analysis_reports_base_dir = os.path.join(script_dir, "Analysis Reports")
    project_name_abs = os.path.abspath(args.path)
    project_name = os.path.basename(project_name_abs) or "root_project"
    if project_name == ".": project_name = os.path.basename(os.getcwd())
    safe_project_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in project_name)
    project_artifact_dir = os.path.join(analysis_reports_base_dir, safe_project_name)
    os.makedirs(project_artifact_dir, exist_ok=True)
    if args.debug: print(f"  [DEBUG Artifacts] Project artifacts in: {project_artifact_dir}")

    llm, embeddings = None, None
    if args.llm_provider == "gemini":
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key: print("Error: GOOGLE_API_KEY not set for Gemini."); sys.exit(1)
        try:
            llm = ChatGoogleGenerativeAI(model=args.model_name, temperature=0, google_api_key=gemini_api_key)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
            if args.debug: llm.invoke("Hello from Gemini!") # Test
        except Exception as e: print(f"Error initializing Gemini: {e}"); sys.exit(1)
    elif args.llm_provider == "custom":
        if not effective_api_key or not effective_endpoint_url:
            print("Error: Custom LLM API key or endpoint URL missing."); sys.exit(1)
        try:
            llm = ChatOpenAI(model=args.model_name, temperature=0, api_key=effective_api_key, base_url=effective_endpoint_url)
            embeddings = OpenAIEmbeddings(model=args.model_name, api_key=effective_api_key, base_url=effective_endpoint_url)
            if args.debug: llm.invoke("Hello from Custom LLM!") # Test
        except Exception as e: print(f"Error initializing custom LLM: {e}"); sys.exit(1)
    else: print(f"Error: Unknown LLM provider '{args.llm_provider}'."); sys.exit(1)

    # Call generate_full_report
    llm_overall_summary, all_docs_for_db, static_findings, llm_results_cache, processed_files_list = generate_full_report(
        code_path=args.path,
        artifact_path=project_artifact_dir,
        llm=llm,
        language=args.language,
        config=config, # Pass the resolved config
        debug=args.debug
    )

    # --- Report Generation (moved out of generate_full_report) ---
    report_output_content = ""
    if args.output_format == "json":
        # llm_results_cache contains {hash: analysis_string} for all processed files (cached or new)
        # We need to map hashes back to file paths for a more useful JSON report.
        # Create a dictionary {filepath: llm_analysis_string}
        detailed_llm_analyses_by_file = {}
        for file_path in processed_files_list:
            file_hash = get_file_hash(file_path) # Re-hash, or pass hashes along
            detailed_llm_analyses_by_file[file_path] = llm_results_cache.get(file_hash, "LLM Analysis not found for this file.")

        report_output_content = json.dumps({
            "overall_llm_summary": llm_overall_summary,
            "detailed_llm_analyses_by_file": detailed_llm_analyses_by_file,
            "static_analysis_findings": static_findings # Already {filepath: {tool: results}}
        }, indent=2)
    elif args.output_format == "html":
        html_content = "<!DOCTYPE html><html><head><title>Comprehensive Code Analysis Report</title>"
        html_content += "<link href=\"../static/prism/prism.css\" rel=\"stylesheet\" />" # Relative path for reports in subdirs
        html_content += "<style>body{font-family:sans-serif;margin:20px;background-color:#f4f4f4;color:#333}h1{color:#333;border-bottom:2px solid #666;padding-bottom:10px}h2{color:#444;margin-top:30px}h3{color:#555;margin-top:20px;border-bottom:1px dashed #ccc;padding-bottom:5px}h4{color:#666;margin-top:15px}.toc{margin-bottom:30px;padding:15px;background-color:#e9e9e9;border-radius:5px}.toc ul{list-style-type:none;padding-left:0}.toc li a{text-decoration:none;color:#007bff}.toc li a:hover{text-decoration:underline}.file-analysis-section{margin-bottom:30px;padding:20px;background-color:#fff;border:1px solid #ddd;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1)}.static-results-block{margin-top:15px;padding:15px;background-color:#f9f9f9;border:1px solid #eee;border-radius:4px}.static-results-block ul{padding-left:20px;margin-top:5px}.static-results-block li{margin-bottom:5px}pre[class*='language-']{padding:1em;margin:.5em 0;overflow:auto;border-radius:.3em;border:1px solid #ddd}</style>"
        html_content += "</head><body>"
        html_content += "<script src=\"../static/prism/prism.js\"></script>"
        html_content += "<script src=\"../static/prism/prism-python.min.js\"></script>"
        html_content += "<script src=\"../static/prism/prism-cpp.min.js\"></script>"
        html_content += "<script src=\"../static/prism/prism-json.min.js\"></script>"
        html_content += "<script src=\"../static/prism/prism-markup.min.js\"></script>" # For Markdown if LLM uses it
        html_content += "<script src=\"../static/prism/prism-markdown.min.js\"></script>"


        html_content += "<h1>Comprehensive Code Analysis Report</h1>"
        html_content += "<div class='toc'><h2>Table of Contents (Analyzed Files)</h2><ul>"
        for i, file_path_report in enumerate(processed_files_list):
            html_content += f"<li><a href='#file-{i}'>{os.path.basename(file_path_report)} ({file_path_report})</a></li>"
        html_content += "</ul></div>"

        html_content += "<h2>Overall LLM Summary</h2>"
        html_content += f"<div class='file-analysis-section'><pre><code class='language-text'>{llm_overall_summary}</code></pre></div>"

        html_content += "<h2>Detailed Analysis Per File</h2>"
        for i, file_path_report in enumerate(processed_files_list):
            file_basename_report = os.path.basename(file_path_report)
            file_hash_report = get_file_hash(file_path_report)
            llm_analysis_for_file_report = llm_results_cache.get(file_hash_report, "LLM analysis not available.")
            static_findings_for_file_report = static_findings.get(file_path_report, {})

            html_content += f"<div id='file-{i}' class='file-analysis-section'>"
            html_content += f"<h3>File: {file_basename_report} <small>({file_path_report})</small></h3>"

            html_content += "<h4>Static Analysis Findings:</h4>"
            if static_findings_for_file_report:
                html_content += "<div class='static-results-block'>"
                for tool, findings_obj in static_findings_for_file_report.items():
                    html_content += f"<h5>{tool.capitalize()} Results:</h5>"
                    if isinstance(findings_obj, list) and findings_obj: # Pylint
                        html_content += "<ul>"
                        for item in findings_obj: html_content += f"<li><b>{item.get('type','issue').capitalize()}</b> (L{item.get('line','?')}): {item.get('message','N/A')} ({item.get('symbol','')})</li>"
                        html_content += "</ul>"
                    elif isinstance(findings_obj, dict) and 'results' in findings_obj: # Cppcheck
                        if findings_obj['results']:
                            html_content += "<ul>"
                            for item in findings_obj['results']: html_content += f"<li><b>{item.get('severity','issue').capitalize()}</b> (L{item.get('locations')[0].get('line','?') if item.get('locations') else '?'}): {item.get('msg','N/A')} ({item.get('id','')})</li>"
                            html_content += "</ul>"
                        else: html_content += "<p>No issues found by Cppcheck.</p>"
                    elif isinstance(findings_obj, dict) and 'error' in findings_obj:
                        html_content += f"<p>Error: <pre><code>{findings_obj['error']}\n{findings_obj.get('details','')}</code></pre></p>"
                    elif not findings_obj: html_content += f"<p>No issues found by {tool}.</p>"
                    else: html_content += f"<p>Raw: <pre><code class='language-json'>{json.dumps(findings_obj, indent=2)}</code></pre></p>"
                html_content += "</div>"
            else: html_content += "<p>No static analysis findings or tools not run.</p>"

            html_content += "<h4>LLM Analysis:</h4>"
            # Use language-markdown for LLM output as it might contain ``` blocks
            html_content += f"<pre><code class='language-markdown'>{llm_analysis_for_file_report}</code></pre>"
            html_content += "</div>"
        html_content += "</body></html>"
        report_output_content = html_content
    else: # Default to text
        report_output_content = f"Overall LLM Summary:\n{llm_overall_summary}\n\n"
        report_output_content += "--- Detailed LLM Analyses Per File ---\n"
        for file_path_report in processed_files_list:
            file_basename_report = os.path.basename(file_path_report)
            file_hash_report = get_file_hash(file_path_report)
            llm_analysis_for_file_report = llm_results_cache.get(file_hash_report, "LLM analysis not available.")
            report_output_content += f"\nFile: {file_basename_report} ({file_path_report})\nLLM Analysis:\n{llm_analysis_for_file_report}\n"

        report_output_content += "\n--- Static Analysis Findings ---\n"
        if static_findings:
            for file_path_report, findings_by_tool in static_findings.items():
                report_output_content += f"\nFile: {os.path.basename(file_path_report)} ({file_path_report})\n"
                for tool, findings_obj in findings_by_tool.items():
                    report_output_content += f"  {tool.capitalize()} Results:\n"
                    if isinstance(findings_obj, list): # Pylint
                        if not findings_obj: report_output_content += "    No issues found.\n"
                        for item in findings_obj: report_output_content += f"    - {item.get('type')}, L{item.get('line')}: {item.get('message')} ({item.get('symbol')})\n"
                    elif isinstance(findings_obj, dict) and 'results' in findings_obj: # Cppcheck
                        if not findings_obj['results']: report_output_content += "    No issues found.\n"
                        for item in findings_obj['results']: loc = item.get('locations')[0] if item.get('locations') else {}; report_output_content += f"    - {item.get('severity')}, L{loc.get('line')}: {item.get('msg')} ({item.get('id')})\n"
                    elif isinstance(findings_obj, dict) and 'error' in findings_obj: report_output_content += f"    Error: {findings_obj['error']}\n"
                    elif not findings_obj: report_output_content += "    No issues found.\n"
                    else: report_output_content += f"    Raw: {json.dumps(findings_obj)}\n"
        else: report_output_content += "No static analysis findings.\n"

    if args.output_file:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base_output_filename = os.path.basename(args.output_file)
        timestamped_filename = f"{timestamp}_{base_output_filename}"
        final_report_path = os.path.join(project_artifact_dir, timestamped_filename)
        try:
            with open(final_report_path, 'w') as f: f.write(report_output_content)
            print(f"\n--- Report Complete: Saved to {final_report_path} ---")
        except IOError as e:
            print(f"Error writing report to {final_report_path}: {e}\nReport:\n{report_output_content}")
    else:
        print("\n--- Report Complete ---")
        print(report_output_content)
        print("--- End of Report ---")

    if all_docs_for_db:
        db = create_vector_store(all_docs_for_db, embeddings, project_artifact_dir, debug=args.debug)
        if db:
            start_qa_session(llm=llm, vector_store=db)
    else:
        print("\nNo documents processed for Q&A vector store.")
