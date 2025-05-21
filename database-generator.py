import os
import chromadb
from chromadb.utils import embedding_functions
import clang.cindex
from typing import List, Dict, Tuple, Optional

# --- Configuration ---
# Path to the code repository you want to process
CODE_REPO_PATH = "/home/fangfufu/projects/httpdirfs/src/"  # <--- CHANGE THIS
# ChromaDB settings
CHROMA_PERSIST_DIRECTORY = "./chroma_db_functions"
CHROMA_COLLECTION_NAME = "code_functions"
# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL
OLLAMA_EMBEDDING_MODEL_NAME = "nomic-embed-text"  # Or your preferred Ollama embedding model

# Supported file extensions for C/C++
SUPPORTED_EXTENSIONS = {".c", ".cpp", ".cc", ".h", ".hpp", ".cxx", ".hxx"}

# --- libclang Configuration ---
# Attempt to find libclang. On some systems, you might need to set this manually.
# Common paths:
# Linux: /usr/lib/x86_64-linux-gnu/libclang-14.so.1 (or similar version)
# macOS (via Homebrew with llvm): /opt/homebrew/opt/llvm/lib/libclang.dylib
# Windows: Path to libclang.dll in your LLVM installation
LIBCLANG_PATH = "/usr/lib/x86_64-linux-gnu/libclang-14.so.1" # Set this if clang.cindex.Config.set_library_file() fails automatically
if LIBCLANG_PATH:
    try:
        clang.cindex.Config.set_library_file(LIBCLANG_PATH)
        print(f"Using libclang from: {LIBCLANG_PATH}")
    except Exception as e:
        print(f"Error setting libclang path: {e}. Ensure libclang is installed and path is correct.")
        print("Consider setting the LIBCLANG_PATH variable in the script.")
        exit(1)
else:
    print("Attempting to find libclang automatically. If this fails, set LIBCLANG_PATH in the script.")


def find_code_files(repo_path: str) -> List[str]:
    """Finds all C/C++ source and header files in the given directory."""
    code_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in SUPPORTED_EXTENSIONS:
                code_files.append(os.path.join(root, file))
    return code_files

def extract_functions_from_file(file_path: str) -> List[Dict[str, str]]:
    """
    Parses a C/C++ file using libclang and extracts functions.
    Returns a list of dictionaries, each containing 'name', 'signature', and 'body', 'full_code'.
    """
    functions = []
    try:
        index = clang.cindex.Index.create()
        # Add common include paths for C++; adjust as needed for your project
        # For complex projects, you might need to pass compiler arguments specific to your build system.
        # Common system include paths might be automatically found, but project-specific ones won't.
        args = ['-x', 'c++' if file_path.endswith(('.cpp', '.hpp', '.cc', '.cxx', '.hxx')) else 'c']
        # Example for adding include paths (you might need to customize this)
        # args.extend(['-I/usr/include', '-I/usr/local/include'])
        # If your repository has an 'include' folder:
        # project_include_path = os.path.join(os.path.dirname(CODE_REPO_PATH), "include") # Or relative to repo_path
        # if os.path.exists(project_include_path):
        #    args.extend([f'-I{project_include_path}'])

        translation_unit = index.parse(file_path, args=args)

        if not translation_unit:
            print(f"Error: Could not parse {file_path}")
            return functions

        for cursor in translation_unit.cursor.walk_preorder():
            if cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                # Only consider function definitions, not just declarations
                if cursor.is_definition():
                    function_name = cursor.spelling
                    function_signature = "" # More complex to reconstruct perfectly without full token iteration
                    function_body = ""
                    full_function_code = ""

                    # Get the source range of the function
                    start_location = cursor.extent.start
                    end_location = cursor.extent.end

                    # Read the function's source code directly from the file
                    # This is often more reliable than trying to reconstruct from tokens for the whole extent
                    if start_location.file and start_location.file.name == file_path:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            f.seek(start_location.offset)
                            full_function_code = f.read(end_location.offset - start_location.offset)

                        # A simple way to get a signature (might not be perfect)
                        # Iterate tokens until '{' for signature
                        tokens = cursor.get_tokens()
                        sig_tokens = []
                        body_started = False
                        for token in tokens:
                            if token.spelling == '{':
                                body_started = True
                                break
                            sig_tokens.append(token.spelling)
                        function_signature = " ".join(sig_tokens).strip()


                        # For body, one could try to get tokens after '{' until matching '}'
                        # but using the full_function_code is usually sufficient for RAG.
                        # Here we use the full extracted code as the main content.

                        if full_function_code:
                            functions.append({
                                "name": function_name,
                                "signature": function_signature, # This is a best effort
                                "full_code": full_function_code,
                                "file_path": file_path,
                                "id": f"{file_path}::{function_name}::{start_location.line}:{start_location.column}"
                            })
                        else:
                            print(f"Warning: Could not extract source for function {function_name} in {file_path}")
    except clang.cindex.LibclangError as e:
        print(f"Libclang error processing {file_path}: {e}")
    except Exception as e:
        print(f"Unexpected error processing {file_path}: {e}")
    return functions


def main():
    print(f"Starting code processing for repository: {CODE_REPO_PATH}")

    # --- Initialize ChromaDB Client ---
    print(f"Initializing ChromaDB with persistence at: {CHROMA_PERSIST_DIRECTORY}")
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
    except Exception as e:
        print(f"Error initializing ChromaDB PersistentClient: {e}")
        print("Ensure ChromaDB is installed correctly and the path is writable.")
        return

    # --- Set up Ollama Embedding Function ---
    print(f"Using Ollama model '{OLLAMA_EMBEDDING_MODEL_NAME}' via {OLLAMA_BASE_URL}")
    try:
        ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            url=OLLAMA_BASE_URL,
            model_name=OLLAMA_EMBEDDING_MODEL_NAME
        )
        print("Ollama embedding function initialized.")
    except Exception as e:
        print(f"Error initializing OllamaEmbeddingFunction: {e}")
        print("Ensure Ollama is running and the model is available.")
        print(f"You can pull a model with: ollama pull {OLLAMA_EMBEDDING_MODEL_NAME}")
        return

    # --- Get or Create Collection ---
    print(f"Getting or creating ChromaDB collection: {CHROMA_COLLECTION_NAME}")
    try:
        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=ollama_ef
            # For very large collections, you might want to add metadata like {"hnsw:space": "cosine"}
            # but the default (L2 squared) is often fine.
        )
        print(f"Collection '{CHROMA_COLLECTION_NAME}' ready.")
    except Exception as e:
        print(f"Error getting or creating ChromaDB collection: {e}")
        return

    # --- Process Files and Add to ChromaDB ---
    code_files = find_code_files(CODE_REPO_PATH)
    if not code_files:
        print(f"No C/C++ files found in {CODE_REPO_PATH}. Check SUPPORTED_EXTENSIONS and path.")
        return

    print(f"Found {len(code_files)} C/C++ files to process.")

    total_functions_added = 0
    for i, file_path in enumerate(code_files):
        print(f"\nProcessing file {i+1}/{len(code_files)}: {file_path}...")
        functions_data = extract_functions_from_file(file_path)

        if not functions_data:
            print(f"No functions extracted from {file_path}.")
            continue

        documents_to_add = []
        metadatas_to_add = []
        ids_to_add = []

        for func_data in functions_data:
            # The document for embedding will be the full source code of the function
            documents_to_add.append(func_data["full_code"])
            metadatas_to_add.append({
                "file_path": func_data["file_path"],
                "function_name": func_data["name"],
                "signature": func_data["signature"], # Potentially useful for context
                # Add other metadata as needed, e.g., start/end line numbers if available
            })
            ids_to_add.append(func_data["id"]) # Unique ID for each function

        if documents_to_add:
            try:
                print(f"Adding {len(documents_to_add)} functions from {file_path} to ChromaDB...")
                collection.add(
                    documents=documents_to_add,
                    metadatas=metadatas_to_add,
                    ids=ids_to_add
                )
                total_functions_added += len(documents_to_add)
                print(f"Successfully added functions from {file_path}.")
            except Exception as e:
                print(f"Error adding functions from {file_path} to ChromaDB: {e}")
        else:
            print(f"No valid function data to add from {file_path}")

    print(f"\n--- Processing Complete ---")
    print(f"Total functions added to ChromaDB collection '{CHROMA_COLLECTION_NAME}': {total_functions_added}")
    # You can verify by querying, e.g.:
    # results = collection.query(query_texts=["example function main"], n_results=2)
    # print("Example query results:", results)

if __name__ == "__main__":
    if not os.path.exists(CODE_REPO_PATH) or not os.path.isdir(CODE_REPO_PATH):
        print(f"Error: CODE_REPO_PATH '{CODE_REPO_PATH}' does not exist or is not a directory.")
        print("Please change the 'CODE_REPO_PATH' variable in the script.")
    else:
        main()