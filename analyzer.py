import os
import sys
import time
import json
import hashlib # Import for file hashing
import threading
import argparse
import concurrent.futures
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.exceptions import OutputParserException
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# --- Caching Functions ---

CACHE_FILE = ".analyzer_cache.json"

def load_cache():
    """Loads the analysis cache from a JSON file."""
    if not os.path.exists(CACHE_FILE):
        return {}
    with open(CACHE_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_cache(cache):
    """Saves the analysis cache to a JSON file."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def get_file_hash(file_path):
    """Calculates the SHA-256 hash of a file's content."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# --- Main Analysis Script ---

def generate_full_report(code_path: str, llm, language: str = "python", debug: bool = False):
    """
    Loads a codebase, analyzes it using a parallelized map-reduce strategy with caching,
    and returns a final comprehensive report and the code chunks.
    """
    print("--- Part 1: Starting Map-Reduce Full Report Generation ---")

    LANGUAGE_MAP = {"python": Language.PYTHON, "csharp": Language.CSHARP, "cpp": Language.CPP}
    GLOB_MAP = {
        "python": ["*.py"],
        "csharp": ["*.cs"],
        "cpp": ["*.cpp", "*.h", "*.hpp"],
    }
    
    if language not in LANGUAGE_MAP:
        print(f"Error: Unsupported language '{language}'.")
        sys.exit(1)

    print(f"\n[Step 1] Scanning for {language} files in '{code_path}'...")
    try:
        from glob import glob
        # Use a set to automatically handle duplicates
        files_to_process_set = set()
        # Correctly walk the directory and apply non-recursive globs
        for dirpath, _, _ in os.walk(code_path):
            for pattern in GLOB_MAP[language]:
                files_to_process_set.update(glob(os.path.join(dirpath, pattern)))
        
        files_to_process = list(files_to_process_set)

        if debug:
            print("\n--- Discovered Files (DEBUG) ---")
            for f_path in files_to_process:
                print(f"  -> {f_path}")
            print("--------------------------------\n")

        if not files_to_process:
            print(f"Warning: No {language} files found in '{code_path}'.")
            return None, []
        print(f"Found {len(files_to_process)} file(s) to analyze.")
    except Exception as e:
        print(f"Error finding documents in '{code_path}': {e}")
        sys.exit(1)

    # Load cache and identify files needing analysis
    cache = load_cache()
    files_to_analyze_live = []
    cached_analyses = []
    all_docs = []

    for file_path in files_to_process:
        # Defensive check to ensure we are only processing files
        if not os.path.isfile(file_path):
            if debug:
                print(f"  [DEBUG] Skipping non-file path: {file_path}")
            continue

        file_hash = get_file_hash(file_path)
        if file_hash in cache:
            print(f"  -> [CACHE] Loading analysis for {os.path.basename(file_path)}")
            cached_analyses.append(cache[file_hash])
            all_docs.extend(TextLoader(file_path).load())
        else:
            print(f"  -> [LIVE]  Queueing analysis for {os.path.basename(file_path)}")
            files_to_analyze_live.append((file_path, file_hash))
            all_docs.extend(TextLoader(file_path).load())

    # Process files that need live analysis
    newly_analyzed_data = {}
    if files_to_analyze_live:
        live_docs = [TextLoader(fp).load()[0] for fp, _ in files_to_analyze_live]
        
        print("\n[Step 2] Splitting new/modified documents into code chunks...")
        splitter = RecursiveCharacterTextSplitter.from_language(language=LANGUAGE_MAP[language], chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(live_docs)
        print(f"Split {len(live_docs)} file(s) into {len(chunks)} chunks.")

        map_template = f"Analyze the following chunk of {language} code: {{text}}"
        map_prompt = PromptTemplate.from_template(map_template)
        
        print(f"\n[Step 3] Running live analysis on {len(chunks)} chunks...")
        start_time = time.time()
        output_parser = StrOutputParser()
        map_chain = map_prompt | llm | output_parser
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Map each future back to its original file hash for correct caching
            future_to_hash = {}
            live_doc_index = 0
            for i, (fp, file_hash) in enumerate(files_to_analyze_live):
                # This logic assumes one document per file, which is true for TextLoader
                doc_chunks = splitter.split_documents([live_docs[i]])
                for chunk in doc_chunks:
                    future = executor.submit(map_chain.invoke, {"text": chunk.page_content})
                    future_to_hash[future] = file_hash

            for i, future in enumerate(concurrent.futures.as_completed(future_to_hash)):
                file_hash = future_to_hash[future]
                try:
                    result = future.result()
                    if file_hash not in newly_analyzed_data:
                        newly_analyzed_data[file_hash] = []
                    newly_analyzed_data[file_hash].append(result)
                    
                    elapsed_time = time.time() - start_time
                    mins, secs = divmod(int(elapsed_time), 60)
                    progress_indicator = f"  -> Mapping... [{mins:02d}:{secs:02d}] Progress: {i + 1}/{len(chunks)} chunks analyzed."
                    sys.stdout.write(progress_indicator + '\r')
                    sys.stdout.flush()
                except Exception as e:
                    print(f"\nError processing a chunk for hash {file_hash}: {e}")

        sys.stdout.write(" " * (len(progress_indicator) + 5) + "\r")
        sys.stdout.flush()
        print("\n  -> Live analysis complete.")

        # Combine chunk analyses for each file and update cache
        for file_hash, analyses in newly_analyzed_data.items():
            combined_analysis = "\n---\n".join(analyses)
            cache[file_hash] = combined_analysis
        save_cache(cache)

    # REDUCE STEP (using all analyses - cached and new)
    print("\n[Step 4] Reducing all analyses into the final report...")
    all_analyses = cached_analyses + ["\n---\n".join(analyses) for analyses in newly_analyzed_data.values()]
    doc_summaries = "\n---\n".join(all_analyses)
    
    reduce_template = "Synthesize these code analyses into a single report... {doc_summaries}"
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = reduce_prompt | llm | StrOutputParser()
    report = reduce_chain.invoke({"doc_summaries": doc_summaries})
    
    print("\n--- Part 1 Complete: Full Analysis Report ---")
    print(report)
    print("--- End of Report ---")
    
    return report, all_docs

def create_vector_store(chunks, embeddings):
    """Creates and persists a vector store from code chunks for semantic search."""
    print("\n--- Part 2: Creating Vector Store for Q&A ---")
    if not chunks:
        print("No code chunks available to create a vector store.")
        return None
    try:
        print("[Step 1] Creating vector store from chunks...")
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./code_db")
        print("Vector store created and saved to './code_db'.")
        return vector_store
    except Exception as e:
        print(f"An error occurred during vector store creation: {e}")
        return None

def start_qa_session(llm, vector_store):
    """Starts an interactive Q&A session about the codebase."""
    print("\n--- Part 3: Interactive Code Q&A ---")
    if not vector_store:
        print("Vector store not available. Cannot start Q&A session.")
        return

    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

    print("\nCode Q&A session started. Ask anything about your code.")
    print("Type 'exit' or 'quit' to end the session.")
    while True:
        query = input("\nAsk a question: ")
        if query.lower() in ['exit', 'quit']:
            print("Exiting Q&A session.")
            break
        try:
            answer = qa_chain.invoke(query)
            print("\nAnswer:")
            print(answer['result'])
        except Exception as e:
            print(f"An error occurred during Q&A: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a codebase with LangChain.")
    parser.add_argument("--path", type=str, default=".", help="The path to the codebase directory.")
    parser.add_argument("--language", type=str, default="python", choices=["python", "csharp", "cpp"], help="The programming language of the codebase.")
    parser.add_argument("--llm-provider", type=str, default="gemini", choices=["gemini", "custom"], help="The LLM provider to use.")
    parser.add_argument("--model-name", type=str, default="gpt-4", help="The model name for a custom LLM provider.")
    parser.add_argument("--endpoint-url", type=str, default=None, help="The API endpoint URL for a custom LLM provider.")
    parser.add_argument("--api-key", type=str, default=None, help="The API key for a custom LLM provider.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to print found file paths.")
    args = parser.parse_args()

    llm, embeddings = None, None

    if args.llm_provider == "gemini":
        print("Using Google Gemini as the LLM provider.")
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            print("Error: GOOGLE_API_KEY environment variable not set.")
            sys.exit(1)
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17", temperature=0, google_api_key=gemini_api_key)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
            llm.invoke("Hello, World!")
        except Exception as e:
            print(f"Error initializing Google Gemini API: {e}")
            sys.exit(1)
            
    elif args.llm_provider == "custom":
        print("Using a custom OpenAI-compatible LLM provider.")
        custom_api_key = os.getenv("CUSTOM_API_KEY") or args.api_key
        custom_endpoint_url = os.getenv("CUSTOM_API_URL") or args.endpoint_url
        
        if not custom_api_key or not custom_endpoint_url:
            print("Error: For custom provider, set CUSTOM_API_KEY/CUSTOM_API_URL or pass --api-key/--endpoint-url.")
            sys.exit(1)
        try:
            llm = ChatOpenAI(model=args.model_name, temperature=0, api_key=custom_api_key, base_url=custom_endpoint_url)
            embeddings = OpenAIEmbeddings(model=args.model_name, api_key=custom_api_key, base_url=custom_endpoint_url)
            llm.invoke("Hello, World!")
        except Exception as e:
            print(f"Error initializing custom LLM API: {e}")
            sys.exit(1)

    summary_report, all_docs = generate_full_report(code_path=args.path, llm=llm, language=args.language, debug=args.debug)

    if all_docs:
        db = create_vector_store(all_docs, embeddings)
        if db:
            start_qa_session(llm=llm, vector_store=db)
    else:
        print("\nCould not initialize Q&A session due to an error in a previous step.")
