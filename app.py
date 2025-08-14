import streamlit as st
import streamlit_antd_components as sac
import os
import shutil
import json
from datetime import datetime
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

# LLM Configuration
LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "http://localhost:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "default-model")

# Embedding Model Configuration
EMBEDDING_API_BASE_URL = os.getenv("EMBEDDING_API_BASE_URL", "http://localhost:1234/v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "not-needed")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "default-embedding-model")

# Paths for data storage
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"
METADATA_PATH = os.path.join(DATA_PATH, "metadata.json")

# --- METADATA HELPER FUNCTIONS ---

def read_metadata():
    """Reads the metadata file and returns a dictionary."""
    if not os.path.exists(METADATA_PATH):
        return {}
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_metadata(data):
    """Writes a dictionary to the metadata file."""
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# --- CORE HELPER FUNCTIONS ---

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

def add_to_knowledge_base(uploaded_files):
    """
    Adds new documents to an existing or new vector knowledge base.
    """
    if not uploaded_files:
        st.warning("请至少上传一个文档。", icon="⚠️")
        return

    # Create data directory if it doesn't exist to store permanent files
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    # Save uploaded files to a temporary directory for processing this batch
    temp_dir = "temp_docs"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    metadata = read_metadata()

    for file in uploaded_files:
        # Save to permanent storage
        perm_path = os.path.join(DATA_PATH, file.name)
        with open(perm_path, "wb") as f:
            f.write(file.getbuffer())

        # Update metadata
        metadata[file.name] = {
            "upload_time": datetime.now().isoformat()
        }

        # Save to temp dir for processing this batch only
        temp_path = os.path.join(temp_dir, file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

    write_metadata(metadata)

    with st.spinner("正在处理新文档，请稍候..."):
        # Load only the new documents from the temporary directory
        documents = load_documents(temp_dir)
        if not documents:
            st.error("无法加载任何新文档，请检查文件格式是否正确。", icon="🚨")
            # Clean up temp dir
            shutil.rmtree(temp_dir)
            return

        # Clean up temp dir after loading
        shutil.rmtree(temp_dir)

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        # Generate unique IDs for each chunk
        ids = [f"{chunk.metadata['source']}_{i}" for i, chunk in enumerate(chunks)]

        # Create embeddings
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            openai_api_base=EMBEDDING_API_BASE_URL,
            openai_api_key=EMBEDDING_API_KEY
        )

        try:
            vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

            # Process documents in batches to avoid overwhelming the server
            batch_size = 16
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]

                if i == 0 and not os.path.exists(CHROMA_PATH):
                     # Create a new vector store with the first batch
                    Chroma.from_documents(
                        documents=batch_chunks,
                        embedding=embeddings,
                        ids=batch_ids,
                        persist_directory=CHROMA_PATH
                    )
                else:
                    # Add subsequent batches to the existing vector store
                    vector_store.add_documents(documents=batch_chunks, ids=batch_ids)

                st.toast(f"已处理 {i + len(batch_chunks)} / {len(chunks)} 个文本片段...")

            vector_store.persist()
            st.session_state.kb_built = True
            st.toast(f"成功处理了 {len(uploaded_files)} 个文档！", icon="✅")
        except Exception as e:
            st.error(f"向知识库添加文档时出错: {e}", icon="🚨")
            # Do not set kb_built to False on a failed addition

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
        st.toast("知识库已成功清除！", icon="✅")
    except Exception as e:
        st.error(f"清除知识库时出错: {e}", icon="🚨")


def delete_from_knowledge_base(filename_to_delete):
    """
    Deletes a specific document and its associated vectors from the knowledge base.
    """
    try:
        # Delete the physical file
        file_path = os.path.join(DATA_PATH, filename_to_delete)
        if os.path.exists(file_path):
            os.remove(file_path)

        # Update metadata
        metadata = read_metadata()
        if filename_to_delete in metadata:
            del metadata[filename_to_delete]
            write_metadata(metadata)

        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            openai_api_base=EMBEDDING_API_BASE_URL,
            openai_api_key=EMBEDDING_API_KEY
        )
        vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

        # Find documents with the matching source metadata
        # Note: The source metadata from loaders is often an absolute path.
        # We need to find all documents whose source *ends with* the filename.
        all_docs = vector_store.get()
        ids_to_delete = []
        for i, metadata in enumerate(all_docs['metadatas']):
            if metadata.get('source', '').endswith(filename_to_delete):
                ids_to_delete.append(all_docs['ids'][i])

        # Delete the found documents from ChromaDB
        if ids_to_delete:
            vector_store.delete(ids=ids_to_delete)
            st.toast(f"已成功从知识库中移除文档: {filename_to_delete}", icon="✅")
        else:
            st.warning(f"在向量存储中未找到与 {filename_to_delete} 关联的数据。", icon="⚠️")

    except Exception as e:
        st.error(f"移除文档时出错: {e}", icon="🚨")


def perform_audit(audit_file):
    """
    Performs the compliance and consistency audit on the uploaded file.
    Generates a structured report using the LLM.
    """
    try:
        with st.spinner("正在执行审核... 这将涉及多次调用AI模型，可能需要一些时间。"):
            # Save and load the audit document
            audit_file_path = os.path.join(DATA_PATH, audit_file.name)
            with open(audit_file_path, "wb") as f:
                f.write(audit_file.getbuffer())

            audit_docs = load_documents(DATA_PATH) # Reload to get the new doc

            if not audit_docs:
                st.error("加载待审核文档失败。", icon="🚨")
                return

            # Initialize embeddings and load the vector store
            embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL_NAME,
                openai_api_base=EMBEDDING_API_BASE_URL,
                openai_api_key=EMBEDDING_API_KEY
            )
            vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            # Initialize the LLM
            llm = ChatOpenAI(
                model=LLM_MODEL_NAME,
                temperature=0.1,
                openai_api_base=LLM_API_BASE_URL,
                openai_api_key=LLM_API_KEY
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
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=50,
                length_function=len
            )
            audit_docs_for_splitting = [doc for doc in audit_docs if os.path.basename(doc.metadata.get('source', '')) == audit_file.name]
            audit_chunks = text_splitter.split_documents(audit_docs_for_splitting)


            analysis_results = []
            progress_bar = st.progress(0, text="准备开始分析...")
            for i, chunk in enumerate(audit_chunks):
                # Update progress bar with a summary of the current chunk
                progress_text = f"正在分析第 {i+1}/{len(audit_chunks)} 部分: “{chunk.page_content[:50]}...”"
                progress_bar.progress((i + 1) / len(audit_chunks), text=progress_text)

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
            progress_bar.progress(1.0, text="正在生成最终报告...")

            SUMMARY_PROMPT = ChatPromptTemplate.from_template(
                """
                **角色**: 您是一位首席合规官，负责撰写最终的审核报告。
                **任务**: 将以下逐块的分析结果综合成一份结构化、专业的报告。

                **独立分析结果**:
                ---
                {analysis_results}
                ---

                **报告生成指令**:
                1.  阅读所有提供的独立分析结果。
                2.  以Markdown格式生成一份最终报告，包含以下四个部分，并使用确切的标题：
                    - `### 总体结论`
                    - `### 合规性分析`
                    - `### 内部一致性分析`
                    - `### 改进建议`
                3.  **总体结论**: 提供一个高度概括的总结。以风险评估开始，使用以下关键词之一：**高风险**、**中风险**、**低风险**或**合规**。
                4.  **合规性分析**: 将所有发现的合规性问题整合成一个无序列表。如果没有问题，请说明。
                5.  **内部一致性分析**: 将所有发现的内部不一致问题整合成一个无序列表。如果没有问题，请说明。
                6.  **改进建议**: 根据发现的问题，为改进制度文件提供可行的建议。
                """
            )

            final_report_chain = SUMMARY_PROMPT | llm | StrOutputParser()
            final_report = final_report_chain.invoke({"analysis_results": "\n".join(analysis_results)})

            st.session_state.report = final_report
    except Exception as e:
        st.error(f"审核过程中发生错误: {e}", icon="🚨")
        st.error("这可能是由于与本地LLM的连接问题或文档格式问题。请检查控制台以获取更多详细信息。", icon="ℹ️")
        st.session_state.report = None


# --- STREAMLIT UI ---

st.set_page_config(page_title="金融合规智能审核助手", layout="wide", page_icon="🛡️")

st.title("🛡️ 金融合规智能审核助手")
st.caption("一个AI驱动的工具，旨在确保您的内部制度符合外部监管要求并保持内部一致性。")

# Initialize session state variables
if "kb_built" not in st.session_state:
    st.session_state.kb_built = os.path.exists(CHROMA_PATH)
if "report" not in st.session_state:
    st.session_state.report = None
if "current_kb_step" not in st.session_state:
    st.session_state.current_kb_step = 0


# Main UI Tabs
selected_tab = sac.tabs([
    sac.TabsItem(label='制度审核', icon='file-search'),
    sac.TabsItem(label='知识库管理', icon='database-add'),
], index=0, align='center', variant='outline')


# --- KNOWLEDGE BASE MANAGEMENT TAB ---
if selected_tab == '知识库管理':
    st.header("知识库管理")

    st.info("在这里，您可以向知识库中增量添加或完全清空参考文档。知识库一旦建立，即可在“制度审核”页面使用。", icon="ℹ️")

    st.subheader("1. 上传新参考文件")
    st.markdown("上传外部监管要求、内部历史制度等文件。这些文件将被**添加**到现有知识库中。")
    uploaded_files = st.file_uploader(
        "选择一个或多个文件",
        type=['pdf', 'docx'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    st.subheader("2. 更新或清空知识库")

    # Use columns for side-by-side buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("添加至知识库", type="primary", use_container_width=True):
            if uploaded_files:
                add_to_knowledge_base(uploaded_files)
                if st.session_state.kb_built:
                    sac.alert(
                        label='成功!',
                        description=f'已成功向上知识库中添加 {len(uploaded_files)} 个文档。',
                        color='success',
                        closable=True,
                        icon=True
                    )
            else:
                sac.alert(
                    label='未上传文件',
                    description='请先选择要添加的参考文件。',
                    color='warning',
                    closable=True,
                    icon=True
                )

    with col2:
        if st.button("清空知识库", type="secondary", use_container_width=True):
            clear_knowledge_base()
            st.rerun()

    st.divider()
    st.subheader("3. 现有知识库文档")

    doc_metadata = read_metadata()

    if doc_metadata:
        sort_by = st.selectbox(
            "排序方式",
            options=["上传时间 (由新到旧)", "文件名 (A-Z)"],
            index=0
        )

        doc_items = list(doc_metadata.items())

        if sort_by == "文件名 (A-Z)":
            doc_items.sort(key=lambda item: item[0])
        else: # Default to sorting by time
            doc_items.sort(key=lambda item: item[1]['upload_time'], reverse=True)

        for doc_name, meta in doc_items:
            upload_time = datetime.fromisoformat(meta['upload_time']).strftime("%Y-%m-%d %H:%M:%S")

            col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
            with col1:
                st.text(doc_name)
            with col2:
                st.text(upload_time)
            with col3:
                if st.button("移除", key=f"remove_{doc_name}", use_container_width=True):
                    delete_from_knowledge_base(doc_name)
                    st.rerun()
    else:
        st.markdown("知识库中尚无文档。")


# --- INSTITUTIONAL AUDIT TAB ---
if selected_tab == '制度审核':
    st.header("制度审核")

    if not st.session_state.kb_built:
        sac.alert(
            label="知识库未就绪",
            description="请先在“知识库管理”标签页中构建知识库。",
            color='warning',
            icon=True
        )
    else:
        st.subheader("1. 上传待审核制度文件")
        st.markdown("上传您需要审核的最新制度文件。")
        audit_file = st.file_uploader(
            "选择一个PDF或DOCX文件",
            type=['pdf', 'docx'],
            accept_multiple_files=False,
            label_visibility="collapsed"
        )

        st.subheader("2. 开始审核")
        if sac.buttons([sac.ButtonsItem(label='开始审核', icon='play-circle-fill', color='primary')], index=None):
            if audit_file:
                st.session_state.report = None # Clear previous report
                perform_audit(audit_file)
            else:
                sac.alert(label='未上传文件', description='请上传一个待审核的文件。', color='warning', closable=True, icon=True)

        # --- DISPLAY AUDIT REPORT ---
        if st.session_state.report:
            st.subheader("审核报告")

            # Parse the report to display components
            report_content = st.session_state.report

            # Extract Overall Conclusion for the alert
            try:
                conclusion_line = report_content.split("### 总体结论")[1].split('\n')[1]
                if "高风险" in conclusion_line:
                    alert_color, alert_icon = 'error', 'shield-exclamation'
                elif "中风险" in conclusion_line:
                    alert_color, alert_icon = 'warning', 'shield-half'
                else: # Low-Risk or Compliant
                    alert_color, alert_icon = 'success', 'shield-check'

                sac.alert(
                    label=f"总体结论: {conclusion_line.replace('**', '')}",
                    color=alert_color,
                    icon=alert_icon,
                    size='lg'
                )
            except IndexError:
                # Could not parse conclusion, show a generic message
                st.info("报告已生成，详情请见下方。")


            # Split report into sections for the collapse component
            try:
                compliance_section = "###" + report_content.split("### 合规性分析")[1].split("### 内部一致性分析")[0]
                consistency_section = "###" + report_content.split("### 内部一致性分析")[1].split("### 改进建议")[0]
                suggestions_section = "###" + report_content.split("### 改进建议")[1]
            except IndexError:
                # Fallback if the LLM didn't follow the format perfectly
                compliance_section = "无法解析合规性分析部分。"
                consistency_section = "无法解析内部一致性分析部分。"
                suggestions_section = "无法解析改进建议部分。"

            with st.expander("合规性分析", expanded=True):
                st.markdown(compliance_section)
            with st.expander("内部一致性分析", expanded=True):
                st.markdown(consistency_section)
            with st.expander("改进建议", expanded=True):
                st.markdown(suggestions_section)
