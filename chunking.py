from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
def create_chunks(file_path):
    with open(file_path,
            "r",
            encoding="utf-8") as file:
        text = file.read()

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=60
    )

    chunks =splitter.split_text(text)
    return chunks

    # for i,chunk in enumerate(chunks):
    #     print(f"chunk{i}")
    #     print()
    #     print(chunk)





