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

# Load the NVIDIA API Key
os.environ['NVIDIA_API_KEY'] = str(os.getenv("NVIDIA_API_KEY"))

# Load the language model (LLM)
llm = ChatNVIDIA(model='meta/llama3-70b-instruct')

# Define judge LLM (the model that evaluates responses)
judge_llm = ChatNVIDIA(model='meta/llama3-70b-instruct')

# Function to load, split, and vectorize PDFs
def vector_embedding():
    if "vectors" not in st.session_state:
        # Create embeddings
        st.session_state.embeddings = NVIDIAEmbeddings()
        # Read in directory
        st.session_state.loader = PyPDFDirectoryLoader("./documents")
        # Load in documents
        st.session_state.docs = st.session_state.loader.load()
        # Split documents into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        # Convert to vectors
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Main UI
st.title("Document Q&A with LLM as Judge")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the given context only. 
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Questions: {input}
    """
)

prompt1 = st.text_input("Enter your question here")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("Vector Store DB is Ready")

if prompt1:
    # Step 1: Generate answer based on documents
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Get answer from the retrieval chain
    initial_answer = retrieval_chain.run(input=prompt1)
    
    st.write("Answer: ", initial_answer)

    # Step 2: Use LLM as Judge to assess the answer
    judge_prompt = ChatPromptTemplate.from_template(
        """
        You are a judge that evaluates the quality of the response. Given the context, answer the following:
        1. Is the answer accurate based on the context?
        2. Does the answer fully address the question?
        3. Provide a score from 1 (poor) to 10 (excellent).
        <context>
        {context}
        </context>
        Answer: {answer}
        """
    )

    # Judge the answer
    judge_input = judge_prompt.format(context=" ".join([doc.page_content for doc in st.session_state.final_documents]), answer=initial_answer)
    judge_response = judge_llm(judge_input)
    
    # Step 3: Display the judge's feedback
    st.write("Judge's Feedback: ", judge_response)
    st.write("Judge's Rating: ", judge_response.split('\n')[2])  # Assuming the score is in the third line
