# Local RAG Engine for Code and Documents

This project is a local RAG (Retrieval-Augmented Generation) application with a graphical user interface built using CustomTkinter. It allows users to create searchable indexes from local repositories of source code and documents. You can then ask complex, natural language questions about your knowledge base using a local Large Language Model (LLM) running via Ollama or LM Studio.

The application features an advanced RAG pipeline to ensure high-quality, relevant answers.

## Key Features ✨

  * **Graphical User Interface**: An easy-to-use GUI for indexing, searching, and configuration.
  * **Hybrid Knowledge Base**: Creates separate, specialized vector indexes for source code and documents for higher retrieval accuracy.
  * **State-of-the-Art Models**: Uses best-in-class open-source models for embeddings (`intfloat/multilingual-e5-large-instruct`), code analysis (`jinaai/jina-embeddings-v2-base-code`), and re-ranking (`Qwen/Qwen3-Reranker-0.6B`).
  * **Advanced RAG Pipeline**: Implements modern techniques to improve quality:
      * **HyDE** (Hypothetical Document Embeddings): Transforms user queries for better conceptual matching.
      * **Cross-Encoder Re-ranking**: Intelligently re-ranks initial search results for maximum relevance.
      * **Syntax-Aware Chunking**: Uses `tree-sitter` to split code along logical boundaries (functions, classes) instead of arbitrary chunks.
      * **Optimized Prompting**: Sends a consolidated and clearly structured prompt to the LLM for more accurate and coherent answers.
  * **Efficient Memory Management**: Employs a JIT (Just-In-Time) and "unload-after-use" strategy to keep VRAM usage at a minimum, allowing large models to run on consumer-grade hardware.
  * **Offline First**: Can be fully operated offline. A setup script helps download all necessary models for local use.
  * **Multi-Language Support**: Automatically detects the query language and instructs the LLM to respond in the same language.
  * **Flexible Backend**: Supports any OpenAI-compatible API, including LM Studio and modern versions of Ollama.
  * **Automated Setup**: Comes with a `start.bat` script for Windows to automate environment creation, dependency installation, and model downloads.

## Architecture 🏛️

The application is split into three core Python files, separating the UI from the underlying logic.

  * `app.py`: Contains all the CustomTkinter GUI code and manages user interaction and threading.
  * `rag_core.py`: Handles the entire indexing pipeline, including file discovery, text extraction, and syntax-aware chunking.
  * `rag_core_query_engine.py`: Contains the `HybridRAGQueryEngine` class, which orchestrates the entire query process (HyDE, retrieval, re-ranking, and final answer generation).

### Query Pipeline

The application follows a multi-stage process to answer a user's question, ensuring high accuracy.

```mermaid
graph TD
    A[User Query] --> B{LLM: HyDE<br>Generate Hypothetical Answer};
    B --> C[Vector Search<br>Retrieve Top 20 Candidates from Code and Doc Indexes];
    C --> D{Cross-Encoder<br><b>Re-rank</b> Candidates based on relevance to Original Query};
    D --> E[Top 7 Best Context Snippets];
    E --> F{LLM: Generation<br><b>Consolidated & Structured Prompt</b><br>Answer question using best snippets};
    F --> G[Formatted Markdown Answer];
```

-----

## 🚀 Setup and Installation

Follow these steps to get the application running.

### Prerequisites

  * Python (version 3.8 - 3.12 recommended).
  * Windows Operating System (for the `.bat` script).
  * A local LLM server like [LM Studio](https://lmstudio.ai/) or [Ollama](https://ollama.com/) running with a loaded model (e.g., Llama 3, Mixtral).

### Instructions

1.  **Place Files**: Ensure all project files (`app.py`, `rag_core.py`, `rag_core_query_engine.py`, `requirements.txt`, `download_models.py`, and `start.bat`) are in the same directory.
2.  **Run the Launcher**: Simply double-click the **`start.bat`** file.

The script will automatically perform the following steps:

  * Check if a Python virtual environment (`.venv`) exists and create one if it doesn't.
  * Activate the environment and install all required libraries from `requirements.txt`.
  * Check if the embedding models exist in a `./models` folder. If not, it will run the `download_models.py` script. This is a one-time download that can take several minutes and requires a few gigabytes of disk space.
  * Launch the GUI application.

-----

## How to Use 📖

### 1\. Indexing Your Data

1.  Open the **Indexing** tab.
2.  Click **"Browse..."** to select the root folder of the project or documents you want to index.
3.  The **Project Name** will be auto-filled based on the folder name. You can change it. This name will be used for the database collections.
4.  Use the checkboxes to select whether you want to **Index Code**, **Index Documents**, or both.
5.  Click **"Start Indexing"**. The progress will be displayed in the log window below. You can click **"Stop"** to interrupt the process.

### 2\. Querying Your Knowledge Base

1.  Switch to the **Querying** tab.
2.  Select your indexed project from the **Project** dropdown menu. If your new project doesn't appear, click **"Refresh"**. The engine will load for the selected project.
3.  Select your running **Backend** (LM Studio or Ollama) and specify the model name you are using (e.g., `lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF`).
4.  The **Use HyDE** and **Use Reranker** checkboxes are enabled by default for the best quality. You can disable them to speed up the query at the cost of potential accuracy.
5.  Type your question in the text box and click **"Send"** or press Enter.
6.  The application will perform the multi-stage RAG process and stream the final, Markdown-formatted answer from the LLM.

### 3\. Settings

You can configure the models and hardware settings used by the application.

1.  Click the **"Settings"** button at the bottom of the window.
2.  In the settings window, you can:
      * Select the **Code Model**, **Documents Model**, and **Reranker Model** from dropdown menus containing high-quality, pre-configured choices.
      * Adjust **Hardware Settings**, including the **Compute Device** (Auto, GPU, CPU) and toggling **Half-Precision (FP16)** for better performance on CUDA devices.
3.  Click **"Save and Reload Engine"**. The application will save your choices in a `config.json` file and reload the query engine with the new settings.