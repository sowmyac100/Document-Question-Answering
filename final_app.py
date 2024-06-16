import streamlit as st
import os 
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS 

from dotenv import load_dotenv
load_dotenv()

#load the NVIDIA API Key
os.environ['NVIDIA_API_KEY'] = str(os.getenv("NVIDIA_API_KEY"))

#load llm
llm = ChatNVIDIA(model = 'meta/llama3-70b-instruct') 

#function to load, split, and vectorize PDFs
def vector_embedding():
    if "vectors" not in st.session_state():
        #create embeddings
        st.session_state.embeddings = NVIDIAEmbeddings()
        #read in directory
        st.session_state.loader = PyPDFDirectoryLoader("./documents")
        #load in documents
        st.session_state.docs = st.session_state.loader.load()
        #split documents into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 700, chunk_overlap = 50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        #convert to vectors
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
st.title("Document Q&A")

prompt  = ChatPromptTemplate.from_template(
"""
Answer the questions based on the given context only. 
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}

"""    
)                            

prompt1 = st.text_input("Enter your question here")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("Vector Store DB is Ready")

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)



