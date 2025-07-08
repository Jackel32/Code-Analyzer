import os
import sys
import time
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

# --- Main Analysis Script ---

def generate_full_report(code_path: str, llm, language: str = "python"):
    """
    Loads a codebase, analyzes it using a parallelized map-reduce strategy,
    and returns a final comprehensive report and the code chunks.
    Defaults to 'python' if no language is specified.
    """
    print("--- Part 1: Starting Map-Reduce Full Report Generation ---")

    # Define language-specific file extensions and splitter configurations
    LANGUAGE_MAP = {
        "python": Language.PYTHON,
        "csharp": Language.CSHARP,
        "cpp": Language.CPP,
    }
    GLOB_MAP = {
        "python": "**/*.py",
        "csharp": "**/*.cs",
        "cpp": "**/*.{cpp,h,hpp}",
    }
    
    if language not in LANGUAGE_MAP:
        print(f"Error: Unsupported language '{language}'. Supported languages are: {list(LANGUAGE_MAP.keys())}")
        sys.exit(1)

    # 1. Load the codebase
    print(f"\n[Step 1] Loading {language} code from '{code_path}'...")
    try:
        loader = DirectoryLoader(path=code_path, glob=GLOB_MAP[language], loader_cls=TextLoader, show_progress=True, use_multithreading=True)
        docs = loader.load()
        if not docs:
            print(f"Warning: No {language} files found in '{code_path}'.")
            return None, []
        print(f"Loaded {len(docs)} document(s).")
    except Exception as e:
        print(f"Error loading documents from '{code_path}': {e}")
        sys.exit(1)

    # 2. Split documents into chunks
    print("\n[Step 2] Splitting documents into code chunks...")
    try:
        splitter = RecursiveCharacterTextSplitter.from_language(language=LANGUAGE_MAP[language], chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        print(f"Split documents into {len(chunks)} chunks.")
    except Exception as e:
        print(f"Error splitting documents: {e}")
        sys.exit(1)

    # 3. Define the prompts
    map_template = """
    The following is a chunk of {language} code:
    ---
    {{text}}
    ---
    Based on this code, please provide a concise analysis covering:
    1. A summary of its primary function or purpose.
    2. Any potential bugs, anti-patterns, or security vulnerabilities.
    3. Any dependencies or modules it imports/uses.
    Analysis:
    """.format(language=language)
    map_prompt = PromptTemplate.from_template(map_template)

    reduce_template = """
    The following are analyses of several different code chunks from the same project.
    ---
    {{doc_summaries}}
    ---
    Please synthesize these individual analyses into a single, cohesive report. The report should include:
    1. A high-level summary of the overall codebase's purpose and architecture.
    2. A consolidated list of the most critical bugs and vulnerabilities found.
    3. A list of all unique dependencies identified across all chunks.
    Final Report:
    """
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    # 4. Run the analysis chain with concurrency
    num_chunks = len(chunks)
    print(f"\n[Step 3] Initializing and running the Map-Reduce chain in parallel...")
    print(f"This will involve {num_chunks} parallel 'map' calls and 1 'reduce' call.")
    
    try:
        start_time = time.time()
        
        # MAP STEP (in parallel)
        output_parser = StrOutputParser()
        map_chain = map_prompt | llm | output_parser
        
        individual_analyses = []
        progress_indicator = ""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(map_chain.invoke, {"text": chunk.page_content}) for chunk in chunks]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    elapsed_time = time.time() - start_time
                    mins, secs = divmod(int(elapsed_time), 60)
                    result = future.result()
                    individual_analyses.append(result)
                    progress_indicator = f"  -> Mapping... [{mins:02d}:{secs:02d}] Progress: {i + 1}/{num_chunks} chunks analyzed."
                    sys.stdout.write(progress_indicator + '\r')
                    sys.stdout.flush()
                except Exception as e:
                    print(f"\nError processing a chunk: {e}")

        sys.stdout.write(" " * (len(progress_indicator) + 5) + "\r")
        sys.stdout.flush()
        print("\n  -> Map step complete. All chunks analyzed.")

        # REDUCE STEP
        print("  -> Reducing... Synthesizing final report.")
        doc_summaries = "\n---\n".join(individual_analyses)
        reduce_chain = reduce_prompt | llm | output_parser
        report = reduce_chain.invoke({"doc_summaries": doc_summaries})

        end_time = time.time()
        actual_duration = end_time - start_time
        act_mins, act_secs = divmod(int(actual_duration), 60)
        time_summary = f"\n\n---\n*Analysis completed in {act_mins} minutes and {act_secs} seconds.*"
        report += time_summary

        print("\n--- Part 1 Complete: Full Analysis Report ---")
        print(report)
        print("--- End of Report ---")
        print(f"\nActual time taken for Step 3: {act_mins} minutes and {act_secs} seconds.")
        
        return report, chunks
    except Exception as e:
        print(f"\nAn unexpected error occurred during chain execution: {e}")
        return None, []

def create_vector_store(chunks):
    """Creates and persists a vector store from code chunks for semantic search."""
    print("\n--- Part 2: Creating Vector Store for Q&A ---")
    if not chunks:
        print("No code chunks available to create a vector store.")
        return None
    try:
        # Initialize the embedding model from Google
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Create the vector store using ChromaDB
        print("[Step 1] Creating vector store from chunks...")
        # The persist_directory will save the embeddings to disk
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory="./code_db"
        )
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

    # Create the RetrievalQA chain. This chain retrieves relevant documents
    # from the vector store and uses them to answer the question.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" is a simple method to "stuff" all retrieved docs into the prompt
        retriever=vector_store.as_retriever()
    )

    print("\nCode Q&A session started. Ask anything about your code.")
    print("Type 'exit' or 'quit' to end the session.")
    while True:
        query = input("\nAsk a question: ")
        if query.lower() in ['exit', 'quit']:
            print("Exiting Q&A session.")
            break
        try:
            # Run the chain with your query and print the result
            answer = qa_chain.invoke(query)
            print("\nAnswer:")
            print(answer['result'])
        except Exception as e:
            print(f"An error occurred during Q&A: {e}")

# Run Part 1 and get the chunks for Part 2
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a codebase with LangChain.")
    parser.add_argument("--path", type=str, default=".", help="The path to the codebase directory.")
    parser.add_argument("--language", type=str, default="python", choices=["python", "csharp", "cpp"], help="The programming language of the codebase.")
    
    # Arguments for selecting LLM provider
    parser.add_argument("--llm-provider", type=str, default="gemini", choices=["gemini", "custom"], help="The LLM provider to use.")
    
    # Arguments for custom OpenAI-compatible provider
    parser.add_argument("--model-name", type=str, default="gpt-4", help="The model name for a custom LLM provider.")
    parser.add_argument("--endpoint-url", type=str, default=None, help="The API endpoint URL for a custom LLM provider.")
    parser.add_argument("--api-key", type=str, default=None, help="The API key for a custom LLM provider.")
    
    args = parser.parse_args()

    llm = None
    embeddings = None

    if args.llm_provider == "gemini":
        print("Using Google Gemini as the LLM provider.")
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            print("Error: GOOGLE_API_KEY environment variable not set. Please set it before running.")
            sys.exit(1)
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17", temperature=0, google_api_key=gemini_api_key)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
            llm.invoke("Hello, World!") # Test connection
        except Exception as e:
            print(f"Error initializing or connecting to Google Gemini API: {e}")
            sys.exit(1)
            
    elif args.llm_provider == "custom":
        print("Using a custom OpenAI-compatible LLM provider.")
        # Prioritize environment variables, then fall back to command-line arguments
        custom_api_key = os.getenv("CUSTOM_API_KEY") or args.api_key
        custom_endpoint_url = os.getenv("CUSTOM_API_URL") or args.endpoint_url
        
        if not custom_api_key or not custom_endpoint_url:
            print("Error: For the custom provider, you must set CUSTOM_API_KEY and CUSTOM_API_URL environment variables, or pass --api-key and --endpoint-url.")
            sys.exit(1)
        try:
            llm = ChatOpenAI(model=args.model_name, temperature=0, api_key=custom_api_key, base_url=custom_endpoint_url)
            embeddings = OpenAIEmbeddings(model=args.model_name, api_key=custom_api_key, base_url=custom_endpoint_url)
            llm.invoke("Hello, World!") # Test connection
        except Exception as e:
            print(f"Error initializing or connecting to the custom LLM API: {e}")
            sys.exit(1)

    summary_report, code_chunks = generate_full_report(code_path=args.path, llm=llm, language=args.language)

    if code_chunks:
        # Pass the initialized embeddings model to the vector store creator
        db = create_vector_store(code_chunks, embeddings)
        if db:
            start_qa_session(llm=llm, vector_store=db)
    else:
        print("\nCould not initialize Q&A session due to an error in a previous step.")
