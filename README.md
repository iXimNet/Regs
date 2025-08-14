# 🛡️ Financial Compliance Intelligent Audit Assistant

This project is a prototype of an AI-powered application designed to assist financial compliance officers. It helps automate the process of auditing new internal policies against external regulations and historical internal documents, ensuring consistency and adherence to standards.

The application leverages a Retrieval-Augmented Generation (RAG) pipeline to provide intelligent analysis and feedback.

## ✨ Features

- **Ant Design UI**: A modern and clean user interface built with Streamlit and the `streamlit-antd-components` library.
- **Knowledge Base Management**: Upload multiple PDF and DOCX documents (e.g., external regulations, historical policies) to create a persistent local knowledge base.
- **Intelligent Document Audit**: Upload a new policy document for a comprehensive audit.
- **Structured Audit Reports**: The AI generates a detailed report that includes:
    - An overall conclusion (e.g., High-Risk, Compliant).
    - A specific analysis of compliance with external regulations.
    - An analysis of consistency with internal historical documents.
    - Actionable suggestions for improvement.
- **Local LLM Integration**: Connects to any OpenAI-compatible local LLM service, ensuring data privacy and control.
- **Clear & Reset**: Easily clear the existing knowledge base to start over from scratch.

## 🛠️ Tech Stack

- **Backend**: Python 3.9+
- **Web Framework**: Streamlit
- **UI Components**: streamlit-antd-components
- **AI/RAG Framework**: LangChain
- **Document Processing**: `pypdf`, `python-docx`
- **Vector Database**: ChromaDB
- **LLM Integration**: `langchain-openai` for connecting to local, OpenAI-compatible APIs.

## 🚀 Setup and Installation

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

### 1. Knowledge Base Management

1.  **Navigate** to the **Knowledge Base Management** tab.
2.  **Upload Files**: Click the file uploader to select and upload your reference documents (PDFs and DOCXs). These can be external regulations or old internal policies.
3.  **Build Knowledge Base**: Once your files are uploaded, click the **"Start Building Knowledge Base"** button. The app will process the files and store them in a local vector database. A success message will appear when it's done.
4.  **Clear Knowledge Base**: If you wish to start over, click the **"Clear Knowledge Base"** button. This will delete all uploaded documents and the created database.

### 2. Institutional Audit

1.  **Navigate** to the **Institutional Audit** tab. (This tab is only usable after a knowledge base has been built).
2.  **Upload Audit File**: Click the file uploader to select the single policy document (PDF or DOCX) you wish to audit.
3.  **Begin Audit**: Click the **"Begin Audit"** button. The application will analyze the document chunk by chunk against the knowledge base. This may take some time.
4.  **Review Report**: Once the audit is complete, a structured report will appear. It will show an overall conclusion and collapsible sections for detailed analysis and suggestions.

## ⚠️ Known Limitations

- **Prototype-Level Prompts**: The prompts used for the RAG chain are general-purpose. For production use, they would require significant tuning and optimization based on specific financial domains and regulatory nuances.
- **Sequential Processing**: The audit process analyzes document chunks one by one. This can be slow for very large documents.
- **LLM Dependency**: The quality of the audit report is highly dependent on the capability of the connected Large Language Model. A more powerful model will yield better results.
- **No Document Management**: The UI does not currently support viewing or deleting individual documents from the knowledge base. You can only clear the entire database.
