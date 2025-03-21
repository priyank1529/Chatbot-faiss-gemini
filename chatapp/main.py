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
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_vertexai import VertexAI
import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
import uvicorn
from google.cloud import aiplatform
from google.auth import default

load_dotenv()
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
app = FastAPI()


credentials, project_id = default()
aiplatform.init(project=project_id, credentials=credentials)
UPLOAD_FOLDER = "uploads"
FAISS_INDEX_PATH = "faiss_index.pkl"
model = VertexAI(model_name="gemini-pro", project=os.getenv("PROJECT_KEY"))


if os.path.exists(FAISS_INDEX_PATH):
    with open(FAISS_INDEX_PATH, "rb") as f:
        faiss_index = pickle.load(f)
else:
    faiss_index = None


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
    global faiss_index, model

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
    context_text = "\n".join([doc.page_content for doc in results])
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question based only on the provided context. "
        "Think step by step before providing a detailed answer.\n"
        "<context> {context} </context>\n"
        "Question: {input}",
        template_format="f-string",  # Ensure correct string formatting
    )
    document_chain = create_stuff_documents_chain(model, prompt)
    retriever = faiss_index.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": query, "context": context_text})
    return {"query": query, "results": response["answer"], "success": True}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
