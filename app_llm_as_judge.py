import os
import time
import hashlib
import torch
import streamlit as st
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

#load env variable
load_dotenv()
os.environ["NVIDIA_API_KEY"] = str(os.getenv("NVIDIA_API_KEY"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#app title
st.title("Document Q&A with LLM as Judge (Top-K, Cached)")

#define a folder fingerprint helper (to detect PDF changes)
def dir_fingerprint(path: str) -> str:
    """Create a stable fingerprint for all files under `path` (names+mtimes+sizes)."""
    hasher = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            try:
                st_ = os.stat(p)
                hasher.update(p.encode("utf-8"))
                hasher.update(str(st_.st_mtime_ns).encode("utf-8"))
                hasher.update(str(st_.st_size).encode("utf-8"))
            except FileNotFoundError:
                pass
    return hasher.hexdigest()

#cache embedding client
@st.cache_resource(show_spinner=False)
def get_embeddings():
    # Heavy client: cache once per session
    return NVIDIAEmbeddings(device=device)

#cache answer LLM
@st.cache_resource(show_spinner=False)
def get_answer_llm():
    # Your answerer (same as before): Llama 3 70B instruct
    return ChatNVIDIA(model="meta/llama3-70b-instruct").to(device)

#cache judge llm
@st.cache_resource(show_spinner=False)
def get_judge_llm():
    # Judge from a different family: Mistral Large 2; keep it deterministic
    # If your tenant uses a different model id, adjust the string below accordingly.
    return ChatNVIDIA(model="mistralai/mistral-large-2", temperature=0).to(device)

#define cached FAISS builder (load, split, add page_num, embed, index)
@st.cache_resource(show_spinner=True)
def build_vectorstore(doc_dir: str, chunk_size=700, chunk_overlap=50, fp: str = ""):
    """
    Build FAISS vector store from PDFs under doc_dir.
    `fp` is the directory fingerprint to bust cache when files change.
    """
    loader = PyPDFDirectoryLoader(doc_dir)
    raw_docs = loader.load()  # returns a list of Document(page_content, metadata)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(raw_docs)  # preserves metadata

    # Ensure page_number is present in metadata (if the loader didn't set it)
    norm_chunks = []
    for d in chunks:
        meta = dict(d.metadata or {})
        if "page_number" not in meta:
            # many loaders put page index as "page" (0-based); normalize to 1-based
            if "page" in meta and isinstance(meta["page"], int):
                meta["page_number"] = meta["page"] + 1
        norm_chunks.append(Document(page_content=d.page_content, metadata=meta))

    embeddings = get_embeddings()
    vectordb = FAISS.from_documents(norm_chunks, embeddings)
    return vectordb

#define cached judge call
@st.cache_data(ttl=1800, show_spinner=False)
def cached_judge(context_text: str, question: str, answer: str) -> str:
    """
    Cache judge outputs for identical (context, question, answer) triples for 30 minutes.
    Avoids re-judging the same thing when users repeat questions.
    """
    judge_llm = get_judge_llm()
    judge_prompt = ChatPromptTemplate.from_template(
        """
        You are a STRICT REFEREE. Evaluate ONLY against <context>.
        Answer the following:
        1) Is the answer accurate based on the context?
        2) Does the answer fully address the question?
        3) Provide a score from 1 (poor) to 10 (excellent).

        <context>
        {context}
        </context>

        Question: {question}
        Answer: {answer}
        """
    )
    judge_input = judge_prompt.format(context=context_text, question=question, answer=answer)
    resp = judge_llm(judge_input)  # returns a string from ChatNVIDIA
    return str(resp)

#set UI controls
DOC_DIR = "./documents"
TOP_K = st.sidebar.number_input("Top-K chunks", min_value=1, max_value=20, value=5, step=1)
CHUNK_SIZE = st.sidebar.number_input("Chunk size", min_value=200, max_value=2000, value=700, step=50)
CHUNK_OVERLAP = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=400, value=50, step=10)

#set-up build/rebuild FAISS to store in session
if st.button("Build / Refresh Vector Index"):
    # Bust cache if files changed (fingerprint key)
    fp = dir_fingerprint(DOC_DIR)
    st.session_state["vectordb"] = build_vectorstore(DOC_DIR, CHUNK_SIZE, CHUNK_OVERLAP, fp)
    st.success("Vector Store is ready.")

# on first run, ensure we have a vector store
if "vectordb" not in st.session_state:
    fp = dir_fingerprint(DOC_DIR)
    st.session_state["vectordb"] = build_vectorstore(DOC_DIR, CHUNK_SIZE, CHUNK_OVERLAP, fp)

#answering prompt
question = st.text_input("Enter your question")

if question:
    vectordb = st.session_state["vectordb"]
    retriever = vectordb.as_retriever(search_kwargs={"k": int(TOP_K)})

    # Retrieve top-k once
    topk_docs = retriever.get_relevant_documents(question)

    # Build answer prompt (using top-k context)
    answer_llm = get_answer_llm()
    answer_prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using ONLY the context. If not answerable from context, say you don't know.
        Provide concise, precise steps/specs and include necessary units.

        <context>
        {context}
        </context>

        Question: {input}
        """
    )
    #pass the Doc list as context
    document_chain = lambda ctx_docs, q: answer_llm.invoke(
        answer_prompt.format(context="\n\n".join([d.page_content for d in ctx_docs]), input=q)
    )

    # Generate answer from ONLY the top-k docs
    t0 = time.time()
    answer_text = document_chain(topk_docs, question)
    t1 = time.time()

    #display generated answer
    st.write("**Answer:**")
    st.write(answer_text)

    #show citations from the same top-k
    st.write("**Found on pages/sections:**")
    pages = sorted({d.metadata.get("page_number") for d in topk_docs if d.metadata})
    if pages:
        st.write(", ".join(f"Page {p}" for p in pages if p is not None))
    else:
        for i, d in enumerate(topk_docs, start=1):
            st.write(f"Chunk #{i}")

    #show answer latency
    st.caption(f"Answer generation latency: {t1 - t0:.2f} s")

    #concatenate top-k text as judge context
    ctx_text = "\n\n".join(d.page_content for d in topk_docs)

    #call cached judge on context, question and answer
    judge_text = cached_judge(ctx_text, question, str(answer_text))

    #display judge feedback
    st.write("**Judge’s Feedback:**")
    st.write(judge_text)

    #display score line 
    #quick score extraction
    try:
        candidates = [ln for ln in str(judge_text).splitlines() if "score" in ln.lower() or "rating" in ln.lower()]
        if candidates:
            st.write("**Judge’s Rating:**", candidates[-1])
    except Exception:
        pass
