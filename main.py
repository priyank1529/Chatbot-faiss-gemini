from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    PDFMinerLoader,
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os
import pickle
import faiss
import numpy as np

# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
app = FastAPI()

UPLOAD_FOLDER = "uploads"
FAISS_INDEX_PATH = "faiss_index.pkl"

if os.path.exists(FAISS_INDEX_PATH):
    with open(FAISS_INDEX_PATH, "rb") as f:
        faiss_index = pickle.load(f)
else:
    faiss_index = None


llm = "LLm Model"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def load_documents(file_path: str, file_type: str):
    if file_type == "text/plain":
        return TextLoader(file_path).load()
    elif file_type == "text/csv":
        return CSVLoader(file_path).load()
    elif file_type == "application/pdf":
        return PyMuPDFLoader(file_path).load()
    elif (
        file_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        return UnstructuredWordDocumentLoader(file_path).load()
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")


# def pad_dynamic_embedding(embedding):
#     """
#     Dynamically updates FAISS index dimensions and ensures all embeddings
#     match the max dimension with zero-padding.
#     """
#     global current_max_dim, index, stored_embeddings

#     current_dim = embedding.shape[1]

#     if current_dim > current_max_dim:
#         print(
#             f"Updating FAISS index from {current_max_dim} to {current_dim} dimensions..."
#         )

#         # Update the max dimension
#         current_max_dim = current_dim

#         # Pad old embeddings to the new size
#         padding = np.zeros(
#             (stored_embeddings.shape[0], current_max_dim - stored_embeddings.shape[1])
#         )
#         stored_embeddings = np.hstack((stored_embeddings, padding))

#         # Recreate FAISS index
#         index = faiss.IndexFlatL2(current_max_dim)
#         index.add(stored_embeddings)  # Re-add previous embeddings

#     # Pad the new embedding
#     padding = np.zeros((embedding.shape[0], current_max_dim - current_dim))
#     embedding_fixed = np.hstack((embedding, padding))

#     # Append to sto red embeddings and add to FAISS
#     stored_embeddings = np.vstack((stored_embeddings, embedding_fixed))
#     index.add(embedding_fixed)

#     return embedding_fixed


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    global faiss_index

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    docs = load_documents(file_path, file.content_type)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    documents = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # Convert new documents to embeddings
    new_faiss_index = FAISS.from_documents(documents, embeddings)

    # If an existing FAISS index is present, merge it with the new one
    if faiss_index is not None:
        faiss_index.merge_from(new_faiss_index)
    else:
        faiss_index = new_faiss_index  # Initialize the FAISS index if none exists

    # Save updated FAISS index
    with open(FAISS_INDEX_PATH, "wb") as f:
        pickle.dump(faiss_index, f)

    return {
        "message": "File stored successfully",
        "filename": file.filename,
        "success": True,
    }


# @app.post("/upload/")
# async def upload_file(file: UploadFile = File(...)):
#     global documents

#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     with open(file_path, "wb") as f:
#         f.write(await file.read())
#     docs = load_documents(file_path, file.content_type)

#     text_data = [doc.page_content for doc in docs]  # Extract text

#     embedding = np.array(model.embed_documents(text_data)).astype("float32")
#     embedding_fixed = pad_dynamic_embedding(embedding)

#     index.add(embedding_fixed)  # Ensure correct shape
#     documents.extend(text_data)  # Store document text

#     with open(FAISS_INDEX_PATH, "wb") as f:
#         pickle.dump(index, f)

#     with open(DOCUMENTS_PATH, "wb") as f:
#         pickle.dump(documents, f)
#     return {
#         "message": "File stored successfully",
#         "filename": file.filename,
#         "success": True,
#     }


@app.post("/search/")
async def search_query(query: str = Form(...)):
    global faiss_index

    if not faiss_index:
        return {
            "message": "FAISS index is not available. Upload a document first.",
            "sucess": False,
        }

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    query_embeddings = embeddings.embed_query(query)
    results = faiss_index.similarity_search_by_vector(query_embeddings, k=5)

    return {"query": query, "results": [docs for docs in results], "success": True}
