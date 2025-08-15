# 🛡️ Financial Compliance Intelligent Audit Assistant

This project is a prototype of an AI-powered application designed to assist financial compliance officers. It helps automate the process of auditing new internal policies against external regulations and historical internal documents, ensuring consistency and adherence to standards.

The application leverages a Retrieval-Augmented Generation (RAG) pipeline to provide intelligent analysis and feedback.

## ✨ Features

- **Ant Design UI**: A modern and clean user interface built with Streamlit and the `streamlit-antd-components` library.
- **Knowledge Base Management**: Upload multiple PDF and DOCX documents (e.g., external regulations, historical policies) to create a persistent local knowledge base.
- **Multi-Stage Intelligent Audit**: Upload a policy document for a comprehensive, multi-stage audit that analyzes both document-level structure and clause-level details.
- **Advanced RAG Pipeline**: The audit engine is now powered by an advanced Retrieval-Augmented Generation pipeline, including:
    - **Document-Level Analysis**: A pre-audit step to check for structural completeness and major omissions.
    - **Query Transformation**: Uses an LLM to rewrite policy clauses into better search queries.
    - **Re-ranking**: Employs a Cross-Encoder model to re-rank search results for higher relevance.
- **Comprehensive & Structured Reports**: The AI generates a detailed report that now includes a dedicated section on structural integrity, in addition to compliance, consistency, and suggestions.
- **Flexible & Maintainable Knowledge Base**:
    - Supports incremental additions.
    - Allows for viewing, sorting, and deep deletion of specific documents.
- **Separated LLM/Embedding Service Configuration**: `.env` file supports distinct endpoints for generation and embedding models, mirroring production environments.

## 🛠️ Tech Stack

- **Backend**: Python 3.9+
- **Web Framework**: Streamlit
- **UI Components**: streamlit-antd-components
- **AI/RAG Framework**: LangChain
- **Document Processing**: `pypdf`, `python-docx`
- **Vector Database**: ChromaDB
- **LLM Integration**: `langchain-openai` for connecting to local, OpenAI-compatible APIs.

## 🚀 Audit Engine Workflow

The application's audit process now follows a sophisticated multi-stage workflow to enhance accuracy and rigor:

1.  **Document-Level Pre-Audit**: When an audit begins, the system first performs a high-level review of the entire document. It uses an LLM to check for structural integrity and completeness against the knowledge base, identifying potentially missing sections.
2.  **Clause-Level Analysis (Chunk by Chunk)**: The system then processes the document chunk by chunk in a highly optimized RAG pipeline:
    a.  **Query Transformation**: Each chunk is rewritten by an LLM into a more effective search query.
    b.  **Retrieval**: The system fetches a broad set of 10 potentially relevant documents from the knowledge base.
    c.  **Re-ranking**: A `CrossEncoder` model re-ranks the 10 documents for relevance, selecting the top 3.
    d.  **Analysis**: The original chunk and the top 3 re-ranked documents are sent to the LLM for detailed compliance and consistency analysis.
3.  **Comprehensive Report Synthesis**: Finally, the findings from both the document-level pre-audit and the detailed clause-level analysis are synthesized into a single, multi-part report.

## 🛠️ Setup and Installation

Follow these steps to get the application running on your local machine.

### 1. Prerequisites

- Python 3.9 or newer.
- An OpenAI-compatible local LLM server. Good options include [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.ai/), or a self-hosted TGI endpoint. The server must expose an API endpoint (e.g., `http://localhost:1234/v1`).

### 2. Clone the Repository

Clone this repository to your local machine:
```bash
# HTTPS
git clone <repository_url>
cd <repository_directory>
```

### 3. Install Dependencies

Install all required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
*Note: Depending on your system, `chromadb` might require C++ build tools. If you encounter installation issues, please consult the [ChromaDB documentation](https://docs.trychroma.com/getting-started).*

### 4. Configure the Application

Configuration is managed using a `.env` file for security and flexibility. This allows you to use different servers or providers for the main Language Model and the Embedding Model.

1.  **Create a `.env` file**: In the root of the project, rename or copy the `.env.example` file to `.env`.
    ```bash
    # On Linux/macOS
    cp .env.example .env
    ```

2.  **Edit the `.env` file**: Open the new `.env` file and customize the variables for both the LLM and the Embedding Model.

    **LLM Service (for analysis and generation):**
    -   `LLM_API_BASE_URL`: The URL for your main language model's API.
    -   `LLM_API_KEY`: The API key for this service.
    -   `LLM_MODEL_NAME`: The identifier for the chat/instruct model you want to use.

    **Embedding Service (for knowledge base vectorization):**
    -   `EMBEDDING_API_BASE_URL`: The URL for your embedding model's API. *This can be the same as the LLM URL or different.*
    -   `EMBEDDING_API_KEY`: The API key for this service.
    -   `EMBEDDING_MODEL_NAME`: The identifier for the text embedding model. **This must be a dedicated embedding model** for the application to function correctly.

### 5. Run the Application

Once the dependencies are installed and the configuration is set, run the application using Streamlit:
```bash
streamlit run app.py
```
The application should now be open in your web browser.

## 📖 How to Use

### 1. 知识库管理 (Knowledge Base Management)

1.  **Navigate** to the "知识库管理" tab to manage your reference documents.
2.  **上传文件 (Upload Files)**: Use the file uploader to select one or more reference documents to add to the knowledge base.
3.  **添加至知识库 (Add to KB)**: Click this button to process the new files and add them to the existing knowledge base.
4.  **查看和移除 (View and Remove)**: The main area lists all documents in the KB, their upload times, and provides a "移除" (Remove) button for each. You can sort this list by name or date.
5.  **清空知识库 (Clear KB)**: This button completely deletes all documents and vectors.

### 2. 制度审核 (Institutional Audit)

1.  **Navigate** to the "制度审核" tab (the default view).
2.  **上传待审文件 (Upload Audit File)**: Select the single policy document you wish to audit.
3.  **开始审核 (Begin Audit)**: Click the button to start the multi-stage audit process. This may take some time.
4.  **查看报告 (Review Report)**: Once complete, a comprehensive report appears with sections for overall conclusions, structural analysis, compliance analysis, consistency, and suggestions.

## ⚠️ Known Limitations

- **Prototype-Level Prompts**: The prompts used for the RAG chain are general-purpose. For production use, they would require significant tuning and optimization based on specific financial domains and regulatory nuances.
- **Sequential Processing**: While the RAG pipeline is now more sophisticated, the clause-by-clause analysis is still sequential and can be slow for very large documents. Implementing asynchronous calls would be the next step for major performance gains.
- **LLM Dependency**: The quality of the audit report is highly dependent on the capability of the connected Large Language Models. More powerful models will yield better results.
- **Hardcoded Models**: The Cross-Encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is currently hardcoded. This could be moved to the `.env` file for more flexibility.
