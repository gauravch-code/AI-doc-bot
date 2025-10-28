# AI Documentation Bot for Python Libraries

Python Version: 3.9+

An intelligent RAG (Retrieval-Augmented Generation) bot using an MCP (Multilayer Perceptron) classifier for source prioritization. Built with Python, Pinecone, OpenAI/Ollama, Scikit-learn, and Streamlit. This bot answers questions about **Pandas**, **NumPy**, and **Scikit-learn** documentation, prioritizing official sources over community examples.

## Features

* **Retrieval-Augmented Generation (RAG):** Finds relevant documentation snippets from a vector database.
* **Source Prioritization:** Uses a trained MLP classifier to predict the source type (API spec, tutorial, community) of retrieved snippets.
* **Intelligent Answering:** Leverages an LLM (OpenAI's GPT or a local Ollama model) with a custom prompt that instructs it to prioritize official documentation sources over community sources (like Stack Overflow).
* **Multi-Library Support:** Currently indexed with documentation from Pandas, NumPy, and Scikit-learn.
* **Interactive UI:** Simple and clean web interface built with Streamlit.

## Tech Stack

* **Backend & ML:** Python 3.9+
* **Web Framework:** Streamlit
* **Vector Database:** Pinecone (Cloud)
* **LLM:** OpenAI API (GPT-4o / GPT-3.5-turbo) or local Ollama (e.g., Llama 3.1)
* **Machine Learning:** Scikit-learn (MLPClassifier, TfidfVectorizer)
* **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
* **Web Scraping:** Requests, BeautifulSoup4
* **Core Libraries:** LangChain (for text splitting)

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    # Activate it:
    # Windows: .\venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up API Keys:**
    * Create a `.env` file in the root directory.
    * Add your API keys:
        ```ini
        PINECONE_API_KEY="your-pinecone-key"
        OPENAI_API_KEY="your-openai-key"
        # (Remove OpenAI key if using Ollama exclusively)
        ```
    * Get keys from [Pinecone](https://app.pinecone.io/) and [OpenAI Platform](https://platform.openai.com/).

5.  **(Optional) Set Up Ollama for Local LLM:**
    * Install Ollama from [ollama.com](https://ollama.com/download).
    * Download a model: `ollama pull llama3.1:8b` (or another model).
    * Ensure Ollama is running in the background.
    * Modify `streamlit_app.py` to use `ChatOllama` instead of `OpenAI` if desired.

## Running the Project

1.  **Scrape Data (First Time Only):**
    * Run the scraper to collect documentation and create `corpus.json`.
    ```bash
    python scraper.py
    ```

2.  **Train Classifier (First Time Only):**
    * Run the training script to create the `.pkl` model files.
    ```bash
    python train_classifier.py
    ```

3.  **Build Vector Database (First Time Only):**
    * Ensure your `PINECONE_API_KEY` is set in `.env`.
    * Run the script to populate your Pinecone index.
    ```bash
    python build_vectordb.py
    ```

4.  **Run the Streamlit App:**
    * Make sure your `.env` file is present and API keys are correct.
    * Ensure your `.pkl` files exist.
    * If using Ollama, make sure it's running.
    ```bash
    streamlit run streamlit_app.py
    ```
    * The app should open automatically in your browser (usually at `http://localhost:8501`).

## Testing

Unit testing for AI applications involving external APIs (Pinecone, OpenAI) and complex ML models can be challenging. Meaningful tests often focus on component integration or require mocking.

* **Current Status:** No formal unit tests are implemented currently.
* **Potential Areas for Unit Tests:**
    * **Scraper:** Test functions that parse specific HTML structures (using saved sample HTML files).
    * **Data Processing:** Test text cleaning or metadata generation logic.
    * **API Interaction (Mocking):** Use libraries like `unittest.mock` or `pytest-mock` to simulate responses from Pinecone and OpenAI APIs. This allows testing the `get_priority_answer` function's logic (like prompt construction) without making actual API calls. For example, you could mock `index.query` to return predefined results and mock `openai_client.chat.completions.create` to check if the correct prompt is being sent.
    * **Classifier Loading:** Test if the `.pkl` files load correctly.

* **Manual Testing:** Thoroughly test the Streamlit app with various queries covering different libraries and complexities. Check if citations are correct and if source prioritization appears logical.

## Future Work

* Add more Python libraries (e.g., Matplotlib, TensorFlow, PyTorch).
* Implement more sophisticated query analysis to detect the target library.
* Allow users to select specific libraries to search within.
* Improve the MCP classifier (e.g., use sentence embeddings instead of TF-IDF, tune hyperparameters).
* Containerize the application using Docker.
* Implement formal unit and integration tests with mocking.
* Deploy the Streamlit app (e.g., using Streamlit Community Cloud).
