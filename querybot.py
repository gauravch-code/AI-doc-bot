import os
import joblib  # To load .pkl files
import torch
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

print("Loading API keys from .env file...")
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise EnvironmentError("API keys for Pinecone or OpenAI not found in .env file.")


print("Loading all models...")
try:
    mcp_brain = joblib.load('mcp_brain.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    print("  > MCP classifier models loaded.")
except FileNotFoundError:
    print("ERROR: Could not find .pkl files. Run train_classifier.py first.")
    exit()


MODEL_NAME = 'all-MiniLM-L6-v2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  > Loading SentenceTransformer ({MODEL_NAME}) on {device}...")
embedding_model = SentenceTransformer(MODEL_NAME, device=device)

INDEX_NAME = "python-docs-ai" 
print(f"Connecting to Pinecone index '{INDEX_NAME}'...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
print("  > Pinecone connection successful.")

print("Connecting to OpenAI...")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
print("  > OpenAI connection successful.")

def get_priority_answer(query_text, top_k=5):
    print(f"\nReceived query: '{query_text}'")

    # --- RAG Step: Retrieve Chunks ---
    print(f"  [RAG] Embedding query and searching Pinecone...")
    query_embedding = embedding_model.encode(query_text).tolist()
    query_response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    matches = query_response['matches']
    if not matches:
        return "I'm sorry, I couldn't find any relevant documents."

    # --- MCP Step: Classify Retrieved Chunks ---
    print("  [MCP] Classifying retrieved chunks for source reliability...")
    retrieved_texts = [match['metadata']['text'] for match in matches]
    texts_tfidf = tfidf_vectorizer.transform(retrieved_texts)
    predicted_labels_numeric = mcp_brain.predict(texts_tfidf)
    predicted_labels_string = label_encoder.inverse_transform(predicted_labels_numeric)
    print(f"  > Predicted sources: {predicted_labels_string}")

    # --- LLM Step: Summarize & Generate Answer ---
    print("  [LLM] Building smart prompt and generating answer...")

    context_str = ""
    for i, (match, label) in enumerate(zip(matches, predicted_labels_string)):
        context_str += f"--- Context {i+1} (Predicted Source: {label}) ---\n"
        context_str += f"URL: {match['metadata']['url']}\n"
        context_str += f"Text: {match['metadata']['text']}\n\n"

    system_prompt = f"""
    You are a helpful expert assistant for Python libraries, including pandas, NumPy, and scikit-learn.
    Your job is to answer the user's question based *only* on the context I provide.

    The context comes from different sources and libraries. Pay attention to the 'Predicted Source' and the 'library' mentioned in the URL or metadata.

    You MUST follow these rules:
    1.  Answer the user's question clearly and concisely.
    2.  Use the "Predicted Source" to judge reliability. You MUST prioritize context from 'api_spec' and 'tutorial' sources.
    3.  Treat 'community' sources (like Stack Overflow) as a last resort or for providing examples, especially if official docs are available.
    4.  If context from 'api_spec' or 'tutorial' contradicts 'community', trust 'api_spec' or 'tutorial'.
    5.  When referencing functions or concepts, try to mention which library they belong to (e.g., "pandas' `merge` function" or "NumPy's `array`").
    6.  Cite your sources by mentioning the URL (e.g., "[Source: .../user_guide/merging.html]").
    7.  If the context is not sufficient to answer the question, just say "I'm sorry, I don't have enough information from the documentation to answer that."
    """

    user_prompt = f"User Question: {query_text}\n\n--- Provided Context ---\n{context_str}\n\nBased on the rules and the context, what is the answer?"

    try:
        chat_completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        final_answer = chat_completion.choices[0].message.content
        return final_answer
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        return "There was an error generating the response."


if __name__ == "__main__":
    print("\n--- AI Documentation Bot Ready ---")
    print("Ask a question about pandas, NumPy, or scikit-learn.")
    print("Type 'quit' or 'exit' to stop.")
    
    while True:
        try:
            # Get input from the user
            user_query = input("\nEnter your query: ")
            
            if user_query.lower() in ['quit', 'exit']:
                print("Exiting bot. Goodbye! 👋")
                break
                
            if not user_query:
                continue

            # Get the answer using your main function
            answer = get_priority_answer(user_query)
            
            # Print the result
            print("\n--- ANSWER ---")
            print(answer)
            print("--------------")

        except EOFError: # Handles Ctrl+D in some terminals
             print("\nExiting bot. Goodbye! 👋")
             break
        except KeyboardInterrupt: # Handles Ctrl+C
            print("\nExiting bot. Goodbye! 👋")
            break