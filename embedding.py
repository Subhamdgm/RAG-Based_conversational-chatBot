from sentence_transformers import SentenceTransformer
from chunking import create_chunks

model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def create_doc_embedding(chunks):

    doc_embedding=model.encode(chunks)
    return doc_embedding

def create_query_embedding(query):
    query_embedding = model.encode(query)
    return query_embedding 
