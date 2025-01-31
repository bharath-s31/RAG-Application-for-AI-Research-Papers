import streamlit as st
import io
import fitz 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS  
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document 

def load_and_process_pdfs(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        file_content = uploaded_file.read()
        file_like_object = io.BytesIO(file_content)
        pdf_document = fitz.open(stream=file_like_object, filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text = page.get_text()
            if text.strip(): 
                documents.append(Document(page_content=text, metadata={"page_number": page_num + 1}))
    return documents

st.title("RAG Application for AI Research Papers")
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = load_and_process_pdfs(uploaded_files)
    
    if not documents:
        st.error("No valid text extracted. Try a different PDF.")
    else:
        text_splitting = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitting.split_documents(documents)

        Ollama_Embeddings = OllamaEmbeddings(model="llama3.2")
        db = FAISS.from_documents(documents, Ollama_Embeddings)

        # Initialize LLM
        llm_1 = Ollama(model="llama3.2")

        # Prompt template
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context. 
        Think step by step before providing a detailed answer. 
        <context>
        {context}
        </context>
        Question: {input}""")
        
        document_chain = create_stuff_documents_chain(llm_1, prompt)
        retriever = db.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Query input
        query = st.text_input("Enter your query")

        if query:
            with st.spinner("Fetching answer..."):
                response = retrieval_chain.invoke({"input": query})
            
            st.write("### Response:")
            st.write(response["answer"] if "answer" in response else response)
