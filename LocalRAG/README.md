# Local RAG for Pizza Restaurant Reviews

This project is a command-line application that uses a local Retrieval-Augmented Generation (RAG) pipeline to answer questions about a pizza restaurant. It leverages a dataset of customer reviews stored in `data.csv`.

## Features

- **Local First:** Runs entirely on your local machine. No API keys needed.
- **LLM Powered:** Uses local Large Language Models (LLMs) through [Ollama](https://ollama.com/).
- **Vector Search:** Employs [ChromaDB](https://www.trychroma.com/) as a local vector store for efficient review retrieval.
- **Orchestration:** The RAG pipeline is built and managed using [LangChain](https://www.langchain.com/).

## Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.8+**
2.  **Ollama:** Follow the installation instructions on the [Ollama website](https://ollama.com/).

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/kaitosuzuki-CS/practice.git
    cd Practice/LocalRAG
    ```

2.  **Create and activate a Python virtual environment:**

    ```bash
    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate

    # For Windows
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Ollama models:**
    This project requires two models from Ollama. Run the following commands in your terminal to pull them:

    ```bash
    # For the main chat model
    ollama pull llama3.1

    # For the embedding model
    ollama pull mxbai-embed-large
    ```

    _Ensure the Ollama application is running before executing these commands._

## Usage

Once the setup is complete, you can run the application:

```bash
python main.py
```

- The first time you run the script, it will create vector embeddings from the `data.csv` file and store them in a local ChromaDB instance in the `chroma_langchain_db` directory. This may take a few moments.
- Subsequent runs will reuse the existing database.
- You will then be prompted to ask questions in the terminal. Type your question and press Enter.
- To quit the application, type `q` and press Enter.

### Example Interaction

```
------------------------
Ask your question (q to quit): what do people think of the pepperoni pizza?


AI Response:
Based on the reviews, the pepperoni pizza is highly regarded. One review mentions it as part of a "great pizza" experience, and another highlights a "good pepperoni pizza". It seems to be a popular and well-liked choice.
```

## Project Structure

- `main.py`: The main entry point for the application. It handles user input and the question-answering chain.
- `vector.py`: Manages the creation of the ChromaDB vector store, generates embeddings from the data, and sets up the retriever.
- `data.csv`: The dataset containing pizza restaurant reviews with columns for `Title`, `Review`, `Rating`, and `Date`.
- `requirements.txt`: A list of the Python packages required for the project.
