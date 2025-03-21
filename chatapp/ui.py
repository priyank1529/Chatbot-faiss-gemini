import streamlit as st
import requests

# API Endpoints
UPLOAD_API_URL = "http://localhost:8000/upload/"
SEARCH_API_URL = "http://localhost:8000/search/"

# Upload UI
st.title("Document Upload")
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])
if uploaded_file is not None:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(UPLOAD_API_URL, files=files)
    st.write(response.json())

# Search UI (Chatbot Style)
st.title("Chatbot Search")
query = st.text_input("Ask something about your documents:")
if st.button("Search"):
    response = requests.post(SEARCH_API_URL, data={"query": query})
    st.write("Response:", response.json())
