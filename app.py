import streamlit as st
import streamlit_antd_components as sac
import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- CONFIGURATION ---
# Load environment variables from .env file
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.getenv("API_KEY", "not-needed")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "default-model")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "default-embedding-model")

# Paths for data storage
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

# --- HELPER FUNCTIONS ---

def load_documents(directory_path):
    """
    Loads documents from the specified directory.
    Supports .pdf and .docx files.
    """
    documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif filename.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {filename}: {e}", icon="🚨")
    return documents

def build_knowledge_base(uploaded_files):
    """
    Builds a vector knowledge base from uploaded files.
    1. Saves files to a local directory.
    2. Loads documents from the directory.
    3. Splits documents into chunks.
    4. Creates embeddings and stores them in ChromaDB.
    """
    if not uploaded_files:
        st.warning("Please upload at least one document.", icon="⚠️")
        return

    # Clear the existing data directory to ensure a fresh start
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)

    # Create the data directory
    os.makedirs(DATA_PATH)

    # Save uploaded files to the data directory
    for file in uploaded_files:
        with open(os.path.join(DATA_PATH, file.name), "wb") as f:
            f.write(file.getbuffer())

    with st.spinner("Processing documents... This may take a moment."):
        # 1. Load documents
        documents = load_documents(DATA_PATH)
        if not documents:
            st.error("Could not load any documents. Please check the file formats.", icon="🚨")
            return

        # 2. Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        # 3. Create embeddings and ChromaDB vector store
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            openai_api_base=API_BASE_URL,
            openai_api_key=API_KEY
        )

        try:
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_PATH
            )
            vector_store.persist() # Ensure data is saved to disk
            st.session_state.kb_built = True
        except Exception as e:
            st.error(f"An error occurred while building the vector store: {e}", icon="🚨")
            st.session_state.kb_built = False

def clear_knowledge_base():
    """
    Deletes the data and vector store directories, effectively clearing the knowledge base.
    """
    try:
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        if os.path.exists(DATA_PATH):
            shutil.rmtree(DATA_PATH)
        st.session_state.kb_built = False
        st.session_state.current_kb_step = 0
        st.session_state.report = None
        st.toast("Knowledge Base cleared successfully!", icon="✅")
    except Exception as e:
        st.error(f"An error occurred while clearing the knowledge base: {e}", icon="🚨")


def perform_audit(audit_file):
    """
    Performs the compliance and consistency audit on the uploaded file.
    Generates a structured report using the LLM.
    """
    try:
        with st.spinner("Performing audit... This involves multiple AI calls and may take some time."):
            # Save and load the audit document
            audit_file_path = os.path.join(DATA_PATH, audit_file.name)
            with open(audit_file_path, "wb") as f:
                f.write(audit_file.getbuffer())

            audit_docs = load_documents(DATA_PATH) # Reload to get the new doc

            if not audit_docs:
                st.error("Failed to load the audit document.", icon="🚨")
                return

            # Initialize embeddings and load the vector store
            embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL_NAME,
                openai_api_base=API_BASE_URL,
                openai_api_key=API_KEY
            )
            vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            # Initialize the LLM
            llm = ChatOpenAI(
                model=LLM_MODEL_NAME,
                temperature=0.1,
                openai_api_base=API_BASE_URL,
                openai_api_key=API_KEY
            )

            # This prompt guides the LLM to act as a compliance officer.
            COMPLIANCE_PROMPT = ChatPromptTemplate.from_template(
                """
                **Role**: You are a professional financial compliance officer.
                **Task**: Analyze the following clause from a new internal policy and check for potential conflicts with the provided external regulatory articles.

                **New Policy Clause**:
                ---
                {clause}
                ---

                **Relevant External Regulatory Articles**:
                ---
                {context}
                ---

                **Analysis Instructions**:
                1.  Carefully compare the "New Policy Clause" with the "Relevant External Regulatory Articles".
                2.  Identify any direct contradictions, subtle inconsistencies, or areas where the policy fails to meet regulatory standards.
                3.  If a conflict is found, clearly state the conflict and cite the specific part of the regulation.
                4.  If the clause is compliant, state that no issues were found.
                5.  Your response must be concise and focused solely on compliance analysis.
                """
            )

            # This prompt guides the LLM to check for internal consistency.
            CONSISTENCY_PROMPT = ChatPromptTemplate.from_template(
                """
                **Role**: You are a senior policy analyst at a financial institution.
                **Task**: Analyze the following clause from a new draft policy and check for inconsistencies with the provided historical internal policies.

                **New Policy Clause**:
                ---
                {clause}
                ---

                **Relevant Historical Internal Policy Articles**:
                ---
                {context}
                ---

                **Analysis Instructions**:
                1.  Compare the "New Policy Clause" with the "Relevant Historical Internal Policy Articles".
                2.  Identify any contradictions, significant deviations, or duplications.
                3.  If an inconsistency is found, describe it clearly and reference the historical policy.
                4.  If the clause is consistent, state that.
                5.  Your response must be concise and focused solely on internal consistency.
                """
            )

            # Split the audit document to analyze it chunk by chunk
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            audit_docs_for_splitting = [doc for doc in audit_docs if os.path.basename(doc.metadata.get('source', '')) == audit_file.name]
            audit_chunks = text_splitter.split_documents(audit_docs_for_splitting)


            analysis_results = []
            for i, chunk in enumerate(audit_chunks):
                st.progress((i + 1) / len(audit_chunks), text=f"Analyzing chunk {i+1} of {len(audit_chunks)}...")

                compliance_chain = ({"context": retriever, "clause": RunnablePassthrough()} | COMPLIANCE_PROMPT | llm | StrOutputParser())
                consistency_chain = ({"context": retriever, "clause": RunnablePassthrough()} | CONSISTENCY_PROMPT | llm | StrOutputParser())

                compliance_result = compliance_chain.invoke(chunk.page_content)
                consistency_result = consistency_chain.invoke(chunk.page_content)

                analysis_results.append(
                    f"### Analysis of Document Section (starting with: '{chunk.page_content[:100]}...')\n\n"
                    f"**Compliance Check:**\n{compliance_result}\n\n"
                    f"**Internal Consistency Check:**\n{consistency_result}\n\n---\n"
                )

            # Final report generation
            st.progress(1.0, text="Synthesizing final report...")

            SUMMARY_PROMPT = ChatPromptTemplate.from_template(
                """
                **Role**: You are a Chief Compliance Officer tasked with creating a final audit report.
                **Task**: Synthesize the following chunk-by-chunk analyses into a single, structured, and professional report.

                **Individual Analysis Results**:
                ---
                {analysis_results}
                ---

                **Report Generation Instructions**:
                1.  Read all the individual analysis results provided.
                2.  Generate a final report in Markdown format with the following four sections, using the exact headers:
                    - `### Overall Conclusion`
                    - `### Compliance Analysis`
                    - `### Internal Consistency Analysis`
                    - `### Improvement Suggestions`
                3.  **Overall Conclusion**: Provide a high-level summary. Start with a risk assessment using one of these keywords: **High-Risk**, **Medium-Risk**, **Low-Risk**, or **Compliant**.
                4.  **Compliance Analysis**: Consolidate all identified compliance issues into a bulleted list. If no issues, state that.
                5.  **Internal Consistency Analysis**: Consolidate all identified inconsistencies into a bulleted list. If no issues, state that.
                6.  **Improvement Suggestions**: Based on the issues found, provide actionable recommendations for improving the policy document.
                """
            )

            final_report_chain = SUMMARY_PROMPT | llm | StrOutputParser()
            final_report = final_report_chain.invoke({"analysis_results": "\n".join(analysis_results)})

            st.session_state.report = final_report
    except Exception as e:
        st.error(f"An error occurred during the audit process: {e}", icon="🚨")
        st.error("This could be due to an issue with the local LLM connection or a problem with the document. Please check the console for more details.", icon="ℹ️")
        st.session_state.report = None


# --- STREAMLIT UI ---

st.set_page_config(page_title="Financial Compliance Assistant", layout="wide", page_icon="🛡️")

st.title("🛡️ Financial Compliance Intelligent Audit Assistant")
st.caption("An AI-powered tool to ensure your internal policies meet regulatory standards and maintain internal consistency.")

# Initialize session state variables
if "kb_built" not in st.session_state:
    st.session_state.kb_built = os.path.exists(CHROMA_PATH)
if "report" not in st.session_state:
    st.session_state.report = None
if "current_kb_step" not in st.session_state:
    st.session_state.current_kb_step = 0


# Main UI Tabs
selected_tab = sac.tabs([
    sac.TabsItem(label='Knowledge Base Management', icon='database-add'),
    sac.TabsItem(label='Institutional Audit', icon='file-search'),
], align='center', variant='outline')


# --- KNOWLEDGE BASE MANAGEMENT TAB ---
if selected_tab == 'Knowledge Base Management':
    st.header("Knowledge Base Management")

    sac.steps(
        items=[
            sac.StepsItem(title='Step 1', description='Upload reference documents (PDF, DOCX)'),
            sac.StepsItem(title='Step 2', description='Process files and build vector store'),
            sac.StepsItem(title='Step 3', description='Knowledge base is ready for auditing'),
        ],
        index=st.session_state.current_kb_step,
        format_func='title',
        placement='vertical'
    )

    st.subheader("1. Upload Reference Files")
    st.markdown("Upload external regulations and historical internal policies. These will form the knowledge base for the audit.")
    uploaded_files = st.file_uploader(
        "Select files",
        type=['pdf', 'docx'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.session_state.current_kb_step = 1

    st.subheader("2. Build or Clear Knowledge Base")

    # Use columns for side-by-side buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Building Knowledge Base", type="primary", use_container_width=True):
            if uploaded_files:
                build_knowledge_base(uploaded_files)
                if st.session_state.kb_built:
                    st.session_state.current_kb_step = 2
                    sac.alert(
                        label='Success!',
                        description='The knowledge base has been successfully built and is ready for use.',
                        color='success',
                        closable=True,
                        icon=True
                    )
            else:
                sac.alert(
                    label='No Files Uploaded',
                    description='Please upload at least one reference document before building the knowledge base.',
                    color='warning',
                    closable=True,
                    icon=True
                )

    with col2:
        if st.button("Clear Knowledge Base", type="secondary", use_container_width=True):
            clear_knowledge_base()
            st.rerun()

    if st.session_state.kb_built:
        st.session_state.current_kb_step = 2
        st.info("Knowledge base is ready. You can proceed to the 'Institutional Audit' tab.", icon="✅")


# --- INSTITUTIONAL AUDIT TAB ---
if selected_tab == 'Institutional Audit':
    st.header("Institutional Audit")

    if not st.session_state.kb_built:
        sac.alert(
            label="Knowledge Base Not Ready",
            description="Please build the knowledge base in the 'Knowledge Base Management' tab first.",
            color='warning',
            icon=True
        )
    else:
        st.subheader("1. Upload Document for Audit")
        st.markdown("Upload the new internal policy document you want to audit.")
        audit_file = st.file_uploader(
            "Select a single PDF or DOCX file",
            type=['pdf', 'docx'],
            accept_multiple_files=False,
            label_visibility="collapsed"
        )

        st.subheader("2. Start the Audit")
        if sac.buttons([sac.ButtonsItem(label='Begin Audit', icon='play-circle-fill', color='primary')], index=None):
            if audit_file:
                st.session_state.report = None # Clear previous report
                perform_audit(audit_file)
            else:
                sac.alert(label='No File Uploaded', description='Please upload a document to audit.', color='warning', closable=True, icon=True)

        # --- DISPLAY AUDIT REPORT ---
        if st.session_state.report:
            st.subheader("Audit Report")

            # Parse the report to display components
            report_content = st.session_state.report

            # Extract Overall Conclusion for the alert
            conclusion_line = report_content.split("### Overall Conclusion")[1].split('\n')[1]
            if "High-Risk" in conclusion_line:
                alert_color, alert_icon = 'error', 'shield-exclamation'
            elif "Medium-Risk" in conclusion_line:
                alert_color, alert_icon = 'warning', 'shield-half'
            else: # Low-Risk or Compliant
                alert_color, alert_icon = 'success', 'shield-check'

            sac.alert(
                label=f"Overall Conclusion: {conclusion_line.replace('**', '')}",
                color=alert_color,
                icon=alert_icon,
                size='lg'
            )

            # Split report into sections for the collapse component
            try:
                compliance_section = "###" + report_content.split("### Compliance Analysis")[1].split("### Internal Consistency Analysis")[0]
                consistency_section = "###" + report_content.split("### Internal Consistency Analysis")[1].split("### Improvement Suggestions")[0]
                suggestions_section = "###" + report_content.split("### Improvement Suggestions")[1]
            except IndexError:
                # Fallback if the LLM didn't follow the format perfectly
                compliance_section = "Could not parse compliance section."
                consistency_section = "Could not parse consistency section."
                suggestions_section = "Could not parse suggestions section."


            sac.accordion(
                items=[
                    sac.AccordionItem(label='Compliance Analysis', icon='check-circle', children=[st.markdown(compliance_section)]),
                    sac.AccordionItem(label='Internal Consistency Analysis', icon='copy', children=[st.markdown(consistency_section)]),
                    sac.AccordionItem(label='Improvement Suggestions', icon='lightbulb', children=[st.markdown(suggestions_section)]),
                ],
                multiple=True,
                open_all=True,
                key="audit_report_accordion"
            )
