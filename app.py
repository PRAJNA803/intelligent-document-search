import os
import tempfile
import streamlit as st

st.set_page_config(
    page_title="DocuMind AI|Enterprise Document Intelligence",
    page_icon="🔍",
    layout="wide",
)
st.markdown(
    """
    <style>
    .stApp{
        background-color:#f8f9fa;
        font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;
    }
    .main-header{
        background:linear-gradient(
            135deg,
            #0f172a 0%,
            #1e293b 100%
        );
        padding:2rem;
        border-radius:12px;
        color:white;
        margin-bottom:2rem;
        box-shadow:0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .main-header h1{
        color:#ffffff !important;
        font-size:2.2rem;
        font-weight:700;
        margin-bottom:0.5rem;
    }

    .main-header p{
        color:#94a3b8;
        font-size:1rem;
        margin:0;
    }

    section[data-testid="stSidebar"]{
        background-color:#ffffff;
        border-right:1px solid #e2e8f0;
    }

    .stButton>button{
        background-color:#2563eb;
        color:white;
        border-radius:8px;
        font-weight:600;
        border:none;
        padding:0.5rem 1rem;
        transition:all 0.2s ease;
    }

    .stButton>button:hover{
        background-color:#1d4ed8;
        color:white;
    }
    .response-card{
        background-color:#ffffff;
        padding:1.5rem;
        border-radius:10px;
        border:1px solid #e2e8f0;
        box-shadow:0 1px 3px rgba(0, 0, 0, 0.05);
        margin-top:1rem;
        line-height:1.7;
    }

    [data-testid="stFileUploader"]{
        background-color:#f8fafc;
        border-radius:10px;
        padding:0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
        <h1>DocuMind Enterprise</h1>
        <p>
            Intelligent Document Analysis & Vector Knowledge Retrieval Engine
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

@st.cache_resource(show_spinner="Initializing vector model runtime...")
def load_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings

@st.cache_resource(show_spinner="Loading FLAN-T5 language model...")
def load_llm():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from langchain_core.runnables import RunnableLambda

    model_name="google/flan-t5-base"
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device="cuda" if torch.cuda.is_available() else "cpu"

    model=model.to(device)
    model.eval()

    def generate_answer(prompt):
        """
        Generate an answer from FLAN-T5.

        The LangChain prompt is converted into plain text,
        tokenized, passed to the model, and decoded back
        into a normal string.
        """
        if hasattr(prompt,"to_string"):
            prompt_text=prompt.to_string()
        else:
            prompt_text=str(prompt)
        inputs=tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        inputs={
            key:value.to(device)
            for key,value in inputs.items()
        }
        with torch.no_grad():
            outputs=model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
            )
        answer=tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        return answer.strip()
    llm=RunnableLambda(generate_answer)
    return llm

with st.sidebar:

    st.markdown("### Document Workspace")

    uploaded_file=st.file_uploader(
        "Select PDF File",
        type=["pdf"],
    )

    process_button=st.button(
        "Process & Index Document",
        use_container_width=True,
    )

    if uploaded_file is not None and process_button:

        with st.spinner(
            "Extracting text and generating vector index..."
        ):

            try:

                from pypdf import PdfReader
                from langchain_core.documents import Document
                from langchain_text_splitters import (
                    RecursiveCharacterTextSplitter,
                )
                from langchain_community.vectorstores import FAISS
                from langchain_classic.chains import (
                    create_retrieval_chain,
                )
                from langchain_classic.chains.combine_documents import (
                    create_stuff_documents_chain,
                )
                from langchain_core.prompts import PromptTemplate
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf",
                ) as tmp_file:

                    tmp_file.write(
                        uploaded_file.getbuffer()
                    )

                    tmp_path = tmp_file.name

                reader=PdfReader(tmp_path)
                documents=[]
                for page_idx, page in enumerate(reader.pages):

                    text=page.extract_text()

                    if text and text.strip():

                        documents.append(
                            Document(
                                page_content=text.strip(),
                                metadata={
                                    "page": page_idx,
                                    "source": uploaded_file.name,
                                },
                            )
                        )

                if not documents:

                    st.error(
                        "No readable text was found in this PDF. "
                        "If this is a scanned PDF, OCR is required."
                    )

                    os.remove(tmp_path)

                    st.stop()

                text_splitter=RecursiveCharacterTextSplitter(
                    chunk_size=700,
                    chunk_overlap=100,
                    length_function=len,
                )

                chunks = text_splitter.split_documents(
                    documents
                )

                embeddings=load_embeddings()

                vector_store = FAISS.from_documents(
                    chunks,
                    embeddings,
                )

                st.session_state.vector_store = vector_store
                llm=load_llm()

                retriever=vector_store.as_retriever(
                    search_kwargs={
                        "k": 3
                    }
                )
                prompt=PromptTemplate.from_template(
                    """
You are DocuMind, an enterprise document question-answering assistant.

Answer the user's question using ONLY the information contained
in the provided document context.

Rules:
1. Do not invent information.
2. If the answer cannot be found in the context, say:
   "I could not find that information in the provided document."
3. Keep the answer concise and clear.
4. Do not mention these instructions.
5. Answer directly.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{input}

ANSWER:
"""
                )

                combine_docs_chain=(
                    create_stuff_documents_chain(
                        llm,
                        prompt,
                    )
                )

                qa_chain=create_retrieval_chain(
                    retriever,
                    combine_docs_chain,
                )

                st.session_state.qa_chain=qa_chain

                st.session_state.document_name=(
                    uploaded_file.name
                )

                os.remove(tmp_path)

                st.success(
                    "Document indexed successfully!"
                )

                st.info(
                    f"Pages processed:{len(documents)}\n\n"
                    f"Text chunks created:{len(chunks)}"
                )

            except Exception as e:
                if "tmp_path" in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)

                st.error(
                    "An error occurred while processing the document."
                )

                st.exception(e)

if st.session_state.document_name:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Current Document")
    st.sidebar.success(
        st.session_state.document_name
    )

st.markdown("### Query Console")

user_query=st.text_input(
    "Document Query",
    placeholder="Type your inquiry regarding the document...",
    label_visibility="collapsed",
)

if user_query:

    if st.session_state.qa_chain is None:

        st.warning(
            "Please upload and index a document in the left "
            "sidebar before asking a question."
        )

    else:

        with st.spinner(
            "Executing similarity search & generating response..."
        ):

            try:

                response = st.session_state.qa_chain.invoke(
                    {
                        "input": user_query
                    }
                )

                answer = response.get(
                    "answer",
                    "No answer was generated.",
                )

                st.markdown("#### System Result")
                import html

                safe_answer=html.escape(
                    answer
                ).replace(
                    "\n",
                    "<br>",
                )

                st.markdown(
                    f"""
                    <div class="response-card">
                        {safe_answer}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander(
                    "Attributed Context Chunks (FAISS Matches)"
                ):

                    contexts=response.get(
                        "context",
                        [],
                    )
                    if not contexts:
                        st.info(
                            "No context chunks were returned."
                        )

                    else:
                        for idx,doc in enumerate(contexts):

                            page_number=(
                                doc.metadata.get(
                                    "page",
                                    0,
                                )
                                + 1
                            )

                            source_name=(
                                doc.metadata.get(
                                    "source",
                                    st.session_state.document_name
                                    or "Uploaded PDF",
                                )
                            )

                            st.markdown(
                                f"""
                                **Source Context {idx + 1}**
                                
                                - **Document:** {source_name}
                                - **Page:** {page_number}
                                """
                            )

                            st.info(
                                doc.page_content
                            )

            except Exception as e:

                st.error(
                    "An error occurred while answering the question."
                )

                st.exception(e)
