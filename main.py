# Imports
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import fitz
import os 
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import subprocess

# Create a FastAPI app instance
app = FastAPI()  

# Redirect root to docs
# For some reason, the root URL doesn't redirect to the docs page automatically, so this workaround is used to redirect it.
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

# CORS (Cross-Origin Resource Sharing) Setup for accessing FastAPI 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder & Model Setup
UPLOAD_FOLDER = "uploaded_docs"             # folder where uploaded files will be stored
os.makedirs(UPLOAD_FOLDER, exist_ok=True)   # create the folder in case it doesn't already exist
model = SentenceTransformer('all-MiniLM-L6-v2')
index = None        # index placeholder
text_chunks = []    # Stores the text chunks extracted from the uploaded document

# Extracting text from pdf
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)  # Open the PDF file
    text = ""
    for page in doc:
        text += page.get_text()  # Extract text from each page and append it
    return text                  # Return the full text extracted from the PDF (for verification purposes only; optional)

# Chunking text into smaller pieces
def chunk_text(text, chunk_size=500):
    words = text.split()  # Split the text into words
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]  # Return number of 500-word chunks

# Document Upload Endpoint
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global index, text_chunks

    # Save uploaded file
    file_location = f"{UPLOAD_FOLDER}/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Extract and Chunk using the predefined functions
    text = extract_text_from_pdf(file_location)
    text_chunks = chunk_text(text)

    # Embed and index
    embeddings = model.encode(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Create a FAISS index using placeholder variable for similarity search
    index.add(np.array(embeddings))

    return {
        "message": "PDF uploaded successfully.",
        "chunks": len(text_chunks),     # Number of chunks created (for verification purposes only; optional)
        "preview": text_chunks[:3]      # Return a preview of the first few chunks (for verification purposes only; optional)
    }

# Chat Functionality Endpoint
class Query(BaseModel):
    question: str

# Using Ollama (LLaMA 2 API) to answer questions based on the uploaded PDF    
@app.post("/chat")
def chat(query: Query):
    global index, text_chunks

    if index is None:
        return {"error": "No document uploaded yet."}

    # Embed user question and find top 3 matches
    q_embedding = model.encode([query.question])        # Convert the question to an embedding
    D, I = index.search(np.array(q_embedding), k=3)     # Search for top 3 similar chunks
    results = [text_chunks[i] for i in I[0]]            # Retrieve the corresponding text chunks

    # Prompt for the API
    prompt = f"Use the following document content to answer the question:\n\n{results}\n\nQuestion: {query.question}\nAnswer:"  # Construct a prompt

    # Run with Ollama (LLaMA 2)
    llama_process = subprocess.run(
        ["ollama", "run", "llama2"],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if llama_process.returncode != 0:
        return {"error": llama_process.stderr.decode("utf-8")}  # (In case of API error, but GPT is locally installed so everything should be fine)

    response = llama_process.stdout.decode("utf-8")     # Decode the output from LLaMA
    return {"answer": response.strip()}                 # Return the result to the user
