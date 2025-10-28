import streamlit as st
import os
import joblib
import torch
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

@st.cache_resource
def load_keys_and_clients():
    load_dotenv()
    pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if not pinecone_api_key or not openai_api_key:
        st.error("API keys missing. Ensure .env file has PINECONE_API_KEY and OPENAI_API_KEY.")
        st.stop() # Stop execution if keys are missing
    
    pc = Pinecone(api_key=pinecone_api_key)
    openai_client = OpenAI(api_key=openai_api_key)
    
    return pc, openai_client

pc, openai_client = load_keys_and_clients()

@st.cache_resource
def load_models():
    try:
        mcp_brain = joblib.load('mcp_brain.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        label_encoder = joblib.load('label_encoder.pkl')

        model_name = 'all-MiniLM-L6-v2'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embedding_model = SentenceTransformer(model_name, device=device)
        
        return mcp_brain, tfidf_vectorizer, label_encoder, embedding_model
    except FileNotFoundError:
        st.error("Model .pkl files not found. Run train_classifier.py first.")
        st.stop()

mcp_brain, tfidf_vectorizer, label_encoder, embedding_model = load_models()


@st.cache_resource
def get_pinecone_index():
    index_name = "python-docs-ai"
    try:
        index = pc.Index(index_name)
        index_stats = index.describe_index_stats()
        st.success(f"Connected to Pinecone index '{index_name}'. Stats: {index_stats}")
        return index
    except Exception as e:
        st.error(f"Failed to connect to Pinecone index '{index_name}': {e}")
        st.stop()

index = get_pinecone_index()

def get_priority_answer(query_text: str, top_k: int = 5):
    """Processes the query, retrieves, classifies, and generates answer."""
    try:
        # --- RAG Step ---
        query_embedding = embedding_model.encode(query_text).tolist()
        query_response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        matches = query_response.get('matches', [])
        if not matches:
            return "I couldn't find relevant documents.", []

        # --- MCP Step ---
        retrieved_texts = [match['metadata']['text'] for match in matches]
        texts_tfidf = tfidf_vectorizer.transform(retrieved_texts)
        predicted_labels_numeric = mcp_brain.predict(texts_tfidf)
        predicted_labels_string = label_encoder.inverse_transform(predicted_labels_numeric).tolist()

        # --- LLM Step ---
        context_str = ""
        for i, (match, label) in enumerate(zip(matches, predicted_labels_string)):
            context_str += f"--- Context {i+1} (Predicted Source: {label}, Library: {match['metadata'].get('library', 'N/A')}) ---\n"
            context_str += f"URL: {match['metadata']['url']}\n"
            context_str += f"Text: {match['metadata']['text'][:500]}...\n\n" # Truncate context

        system_prompt = f"""
        You are a helpful expert assistant for Python libraries (pandas, NumPy, scikit-learn).
        Answer the user's question based *only* on the context provided.
        Pay attention to 'Predicted Source' and 'Library'. Prioritize 'api_spec' and 'tutorial' sources.
        Treat 'community' sources as examples or last resort. Trust official docs over community if conflicting.
        Mention the library (e.g., "pandas' `merge`", "NumPy's `array`"). Cite source URLs.
        If context is insufficient, say so. Keep answers concise and format nicely for display.
        """
        user_prompt = f"User Question: {query_text}\n\n--- Provided Context ---\n{context_str}\n\nBased ONLY on the context, what is the answer?"

        chat_completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        final_answer = chat_completion.choices[0].message.content
        return final_answer, predicted_labels_string

    except Exception as e:
        st.error(f"Error during query processing: {e}")
        return "An error occurred while processing your query.", []


st.set_page_config(page_title="AI Documentation Bot", layout="wide")
st.title("📚 AI Documentation Bot")
st.caption("Ask questions about Pandas, NumPy, and Scikit-learn documentation.")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question here..."):
    
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking... 🤔"):
        response, sources = get_priority_answer(prompt)
        
        
        full_response = f"{response}\n\n*Retrieved sources (predicted): {sources}*"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})