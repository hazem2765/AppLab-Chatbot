# AppLab Chatbot Assignment

This project is a document-based question answering system built as part of a technical assignment. It uses a combination of FastAPI, Streamlit, and LLaMA 2 (via Ollama) to provide a conversational interface for querying the contents of uploaded PDF documents.

## Overview

The application allows a user to:

1. Upload a PDF file.
2. Automatically extract and chunk the text from the document.
3. Convert the chunks into semantic embeddings using a SentenceTransformer model.
4. Store those embeddings in a FAISS index for fast similarity search.
5. Accept user questions, retrieve relevant document chunks, and pass the context to a locally running LLaMA 2 model.
6. Return the model’s generated answer to the user in natural language.

## Project Structure

### `main.py`
This is the FastAPI backend. It handles:

- Accepting and processing PDF uploads.
- Embedding document text into vectors.
- Performing semantic similarity search using FAISS.
- Constructing the prompt and communicating with the LLaMA model through Ollama.

### `app.py`
This is the Streamlit-based frontend. It provides a user-friendly interface to:

- Upload PDF files.
- View a preview of extracted text chunks.
- Ask questions and display the answers.

It communicates with the FastAPI server via HTTP requests to the defined endpoints.

### `Launcher.bat`
A Windows batch script to simplify startup. It:

1. Opens a terminal to launch the FastAPI server.
2. Opens a second terminal to launch the Streamlit app.
3. Automatically opens the Streamlit interface in the default browser.

## Setup

To run this application locally, ensure the following are installed:

- Python 3.9+
- Ollama (for running LLaMA 2 locally)
- Git

Install dependencies using:

```bash
pip install -r requirements.txt
