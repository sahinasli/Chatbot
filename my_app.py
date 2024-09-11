import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
from transformers import pipeline
import math

# Load environment variables
load_dotenv()

# Get API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Create OpenAI model
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name='gpt-4o-mini')

# Create LLMChain for asking questions
qa_prompt = PromptTemplate(input_variables=["question", "docs_content"], template="Based on the following content:\n{docs_content}\n\nQ: {question}\nA:")
qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

# Create LLMChain for summarization
summarize_prompt = PromptTemplate(input_variables=["docs_content"], template="Summarize the following content:\n{docs_content}\n\nSummary:")
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

# Streamlit application
st.title("RAG Chatbot with PDF Summarization")
st.write("Upload a PDF, ask questions, and get summaries!")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_text(text)

    # Initialize embeddings and Chroma vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma.from_texts(docs, embeddings)

    # Select mode: Question-Answering or Summarization
    mode = st.selectbox("Choose a mode:", ["Question-Answering", "Summarization"])

    if mode == "Question-Answering":
        # Question-Answering mode
        question = st.text_input("Ask a question based on the PDF:")
        if question:
            relevant_docs = vectorstore.similarity_search(question)
            docs_content = "\n".join([doc.page_content for doc in relevant_docs])
            answer = qa_chain.run(question=question, docs_content=docs_content)
            st.write("Answer:", answer)

    elif mode == "Summarization":
        # Summarization mode
        docs_content = "\n".join(docs)
        summary = summarize_chain.run(docs_content=docs_content)
        st.write("Summary:", summary)

# Use Hugging Face pipeline for summarization
summarizer = pipeline("summarization")

def split_text(text, max_chunk_length):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_chunk_length):
        chunk = ' '.join(words[i:i + max_chunk_length])
        chunks.append(chunk)
    return chunks

def summarize_chunks(chunks, max_length=130, min_length=30):
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return summaries

# Example of a long text
long_text = """
Place the long text you want to summarize here.
"""

# Split text into chunks not exceeding 4097 tokens
chunks = split_text(long_text, max_chunk_length=4097)

# Summarize each chunk
summaries = summarize_chunks(chunks)

# Combine summaries
final_summary = ' '.join(summaries)

print("Summarized Text:")
print(final_summary)
