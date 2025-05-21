import chromadb
from chromadb.utils import embedding_functions
import ollama # For interacting with Ollama LLMs
import os

# --- Configuration (should match your ingestion script where applicable) ---
# ChromaDB settings
CHROMA_PERSIST_DIRECTORY = "./chroma_db_functions" # Path where your DB was saved
CHROMA_COLLECTION_NAME = "code_functions"   # Collection name used during ingestion

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"      # Default Ollama URL
OLLAMA_EMBEDDING_MODEL_NAME = "nomic-embed-text" # **MUST** be the same model used for ingestion
OLLAMA_GENERATION_MODEL_NAME = "gemma3:12b" # Or "mistral", "codellama", or any other Ollama model suitable for generation

# RAG settings
TOP_N_RESULTS = 5 # Number of relevant code snippets to retrieve

def main():
    # --- Initialize ChromaDB Client ---
    if not os.path.exists(CHROMA_PERSIST_DIRECTORY):
        print(f"Error: ChromaDB persistence directory '{CHROMA_PERSIST_DIRECTORY}' not found.")
        print("Please ensure you have run the ingestion script first and the path is correct.")
        return

    print(f"Connecting to ChromaDB at: {CHROMA_PERSIST_DIRECTORY}")
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
    except Exception as e:
        print(f"Error initializing ChromaDB PersistentClient: {e}")
        return

    # --- Set up Ollama Embedding Function ---
    # This is needed to embed the user's query
    print(f"Initializing Ollama embedding function with model '{OLLAMA_EMBEDDING_MODEL_NAME}' via {OLLAMA_BASE_URL}")
    try:
        ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            url=OLLAMA_BASE_URL,
            model_name=OLLAMA_EMBEDDING_MODEL_NAME
        )
    except Exception as e:
        print(f"Error initializing OllamaEmbeddingFunction: {e}")
        print(f"Ensure Ollama is running and the model '{OLLAMA_EMBEDDING_MODEL_NAME}' is available.")
        return

    # --- Get the Collection ---
    print(f"Accessing ChromaDB collection: {CHROMA_COLLECTION_NAME}")
    try:
        collection = client.get_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=ollama_ef # Provide the EF for query embedding
        )
        print(f"Collection '{CHROMA_COLLECTION_NAME}' accessed successfully.")
    except Exception as e:
        # Handle case where collection might not exist, though get_collection should raise if it doesn't
        print(f"Error accessing collection '{CHROMA_COLLECTION_NAME}': {e}")
        print("Ensure the collection was created by the ingestion script.")
        return

    # --- Initialize Ollama client for generation ---
    try:
        ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
        ollama_models_response = ollama_client.list() # This should return a dict like {'models': [...]}

        available_model_names = []
        if 'models' in ollama_models_response and isinstance(ollama_models_response['models'], list):
            for model_info in ollama_models_response['models']:
                if isinstance(model_info, dict):
                    # The 'name' field is usually what identifies the model tag (e.g., "llama3:latest").
                    model_name = model_info.get('name')
                    if model_name:
                        available_model_names.append(model_name)
                    else:
                        # If 'name' is missing, print a warning but continue.
                        # You could also try model_info.get('model') as a fallback if necessary.
                        print(f"Warning: An Ollama model entry was found without a 'name' key. Details: {model_info}")
                else:
                    print(f"Warning: Unexpected non-dictionary item found in Ollama models list: {model_info}")
        else:
            print("Warning: Could not retrieve a valid list of models from Ollama.")
            print(f"Raw response from ollama_client.list(): {ollama_models_response}")

        # Check if the desired generation model is available
        generation_model_is_available = False
        if available_model_names: # Only check if we have a list of names
            for m_name in available_model_names:
                # OLLAMA_GENERATION_MODEL_NAME could be a base name like "llama3" or a full tag "llama3:latest".
                # Model names from Ollama list are typically full tags.
                if OLLAMA_GENERATION_MODEL_NAME == m_name or \
                   m_name.startswith(OLLAMA_GENERATION_MODEL_NAME + ":"):
                    generation_model_is_available = True
                    break
        
        if not generation_model_is_available:
            print(f"Warning: Ollama generation model '{OLLAMA_GENERATION_MODEL_NAME}' not found among available models.")
            if available_model_names:
                print(f"Available models from Ollama: {', '.join(available_model_names)}")
            else:
                print("No models were successfully listed from Ollama, or the list was empty/malformed.")
            print(f"Consider pulling the model using: ollama pull {OLLAMA_GENERATION_MODEL_NAME}")
            # Depending on desired behavior, you might want to exit here.
            # The script will proceed, and ollama.chat() will likely fail if the model is truly not usable.

    except Exception as e:
        print(f"Error initializing Ollama client or listing models: {e}")
        print(f"Ensure Ollama is running at {OLLAMA_BASE_URL} and is accessible.")
        return # Critical error, cannot proceed

    target_function_name = "Meta_read"  # <--- CHANGE THIS

    print(f"\n--- Attempting to directly GET function: {target_function_name}  ---")
    try:
        # Note: ChromaDB's `get` with a `where` filter can be a bit nuanced.
        # It's often easier if you know the exact ID.
        # If you have an exact ID like "path/to/file.c::function_name::line:col":
        # results_get = collection.get(ids=["exact_id_of_function"])

        # If filtering by metadata:
        # This will retrieve ALL segments that match the where clause.
        # You might need to be more specific if names are not unique across your criteria.
        # We also need to make sure 'is_static' was ingested as metadata.
        get_results = collection.get(
            where={"function_name": target_function_name},
            include=["documents", "metadatas"]
        )

        if get_results and get_results['ids']:
            print(f"Direct GET successful. Found {len(get_results['ids'])} match(es).")
            for i, doc_id in enumerate(get_results['ids']):
                print(f"  ID: {doc_id}")
                print(f"  Metadata: {get_results['metadatas'][i]}")
                print(f"  Document (code):\n```c\n{get_results['documents'][i]}\n```")
                print("-" * 20)
        else:
            print(f"Direct GET failed to find the function with specified criteria.")
            print(f"Consider checking exact metadata values in your SQLite DB for this function.")

    except Exception as e:
        print(f"Error during direct GET: {e}")
    print("--- END OF DIRECT GET TEST ---\n")

    # --- Main RAG Loop ---
    while True:
        user_query = input("\nAsk a question about the C/C++ code (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        if not user_query.strip():
            continue

        print(f"\nEmbedding your query: '{user_query}'")
        # Query ChromaDB for relevant documents (function code snippets)
        # The embedding of the query_texts is handled automatically by ChromaDB
        # when an embedding_function is associated with the collection.
        try:
            results = collection.query(
                query_texts=[user_query],
                n_results=TOP_N_RESULTS,
                include=["documents", "metadatas"] # We want the function code and its metadata
            )
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            continue

        retrieved_documents = results.get('documents', [[]])[0]
        retrieved_metadatas = results.get('metadatas', [[]])[0]

        if not retrieved_documents:
            print("No relevant functions found in the database for your query.")
            continue

        # --- Construct the prompt for the LLM ---
        context_str = "\n\n--- Relevant C/C++ Functions ---\n"
        for i, (doc, meta) in enumerate(zip(retrieved_documents, retrieved_metadatas)):
            context_str += f"\nFunction {i+1} (from {meta.get('file_path', 'N/A')} - {meta.get('function_name', 'N/A')} ):\n"
            context_str += "```cpp\n" # Assuming C++ highlighting for markdown
            context_str += doc
            context_str += "\n```\n"
        context_str += "\n--- End of Relevant Functions ---\n"

        prompt = f"""You are a helpful C/C++ programming assistant.
Based on the following C/C++ function(s) provided as context, answer the user's question.
If the context does not contain the answer, say that you cannot answer based on the provided information.
Do not make up information if it's not in the context.

{context_str}

User's Question: {user_query}

Answer:
"""
        print("\n--- Sending the following prompt to Ollama LLM ---")
        # print(prompt) # Uncomment to see the full prompt
        print(f"Context includes {len(retrieved_documents)} retrieved functions.")
        print(f"User Question: {user_query}")
        print(prompt)
        print("--- Waiting for Ollama's response... ---")

        # --- Query Ollama's LLM for generation ---
        try:
            response = ollama_client.chat(
                model=OLLAMA_GENERATION_MODEL_NAME,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                    }
                ]
            )
            answer = response['message']['content']
            print("\n--- Ollama's Answer ---")
            print(answer)
            print("-----------------------\n")

        except Exception as e:
            print(f"Error querying Ollama generation model '{OLLAMA_GENERATION_MODEL_NAME}': {e}")
            print("Ensure the model is running and available in Ollama.")

    print("Exiting RAG query program.")

if __name__ == "__main__":
    main()