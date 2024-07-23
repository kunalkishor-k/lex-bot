#Loading Library
from dotenv import load_dotenv
import os
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain import llm_cache
from langchain.globals import set_llm_cache


# Load environment variables
load_dotenv()

# Configure the generative AI modelv
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit page configuration
st.set_page_config(
    page_title="KKPDF",
    page_icon=":books:",  # You can use an emoji or a path to an image file
    layout="wide",    # Can be "centered" or "wide"
    initial_sidebar_state="auto"  # Can be "auto", "expanded", "collapsed"
)

# Define the generative model
model = genai.GenerativeModel("gemini-pro")

# Function for extracting text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function for splitting the extracted text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def load_faiss_index(pickle_file):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # Load FAISS index with dangerous deserialization enabled
    faiss_index = FAISS.load_local(pickle_file, allow_dangerous_deserialization=True,embeddings=embeddings)
    return faiss_index

# Function to get vector store
# def get_vector_store(text_chunks):
#     vector_store = Chroma.from_texts(text_chunks, embedding=embeddings,persist_directory="./chroma_db")
#     vector_store.save("Chroma_index")

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    # return vector_store    

# Creating PromptTemplate
prompt_templates = """
Answer the questions in as much detail as possible based on the provided context. 
Ensure that your answers align with the given context. 
If the context is unclear or insufficient, do not provide incorrect or assumed answers. 
Instead, specify the exact information you need to answer the queries accurately. If necessary, respond with thanks.

Additionally, if definitions, key terms, or examples related to the context are requested, please provide them.
\n\n 
Context:\n{context}?\n
Question:\n{question}\n
Answer:
"""

# Function to get conversational chain
def get_conversational_chain():
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5) #temperature=0.3
    prompt = PromptTemplate(template=prompt_templates, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input, similarity search, and getting response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embeddings-001")
    new_db = load_faiss_index("faiss_index")
    # new_db = Chroma(persist_directory="./chroma_db",embedding_function= embeddings)
    docs = new_db.similarity_search(user_question) ## need to learnt he correct syntax

    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question})
    
    st.write("**Reply:**", response["output_text"])

# Streamlit app
def main():
    # st.header("Chat with PDF using Gemini")
    st.header("LexAdvise Co-Pilot")
    user_question = st.text_input("Ask a Question")

    if user_question:
        # vector_store=get_vector_store(text_chunks)
        user_input(user_question)

     # Add temperature slider to sidebar
    temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=1.0, step=0.1, value=0.3)


    with st.sidebar:
        # st.title("Menu:")
        st.title("Upload Pdf")
        pdf_docs = st.file_uploader("Upload your Files and Click Submit", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit"):
            with st.spinner("Uploading..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store=get_vector_store(text_chunks)
                st.success("VectorDB Upload Finished")

# # Load Chroma with dangerous deserialization allowed (only if safe)
# def load_chroma_with_deserialization(pickle_file):
#     return Chroma.load(pickle_file, allow_dangerous_deserialization=True)


if __name__ == "__main__":
    main()
