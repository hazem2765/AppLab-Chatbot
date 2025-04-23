# Imports
import streamlit as st
import requests

# FastAPI endpoint URL
API_URL = "http://127.0.0.1:8000"  # URL of the FastAPI server (Gets redirected in main.py file)

# Allowing Streamlit to accept uploads and send them to FastAPI backend
def upload_pdf(file):
    files = {"file": file.getvalue()}  # Prepare the file as bytes for upload
    response = requests.post(f"{API_URL}/upload", files=files)  # POST request to the /upload endpoint
    return response.json()  # Return the JSON response

# Allowing Streamlit to send questions to FastAPI backend for LLaMA chat
def chat_with_llama(question):
    # Send the question to FastAPI backend for LLaMA chat
    response = requests.post(f"{API_URL}/chat", json={"question": question})  # POST request to /chat endpoint with the question
    return response.json()  # Return the JSON response

# Streamlit UI
st.title("LLaMA-Powered Chatbot")
st.subheader("Upload a PDF Document")

# File Uploader Widget
uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])  # PDF only restriction (as per assignment instructions; can be changed)
if uploaded_file is not None:
    st.write("Uploading and processing your document...")
    response = upload_pdf(uploaded_file)
    if "error" in response:
        st.error(f"Error: {response['error']}")
    else:
        st.success(f"Document uploaded successfully! Number of chunks: {response['chunks']}")
        st.write("Document Preview:")  # Show preview header
        st.write(response['preview'])  # Display the first few chunks of the document

# Chat Frontend Functionality
st.subheader("Ask a question")
question = st.text_input("Your Question:")
if question:
    st.write("Asking LLaMA...")
    answer = chat_with_llama(question)
    if "error" in answer:
        st.error(f"Error: {answer['error']}")
    else:
        st.write(f"**Answer:** {answer['answer']}")
