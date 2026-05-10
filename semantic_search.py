from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from chunking import create_chunks
from embedding import (create_doc_embedding,create_query_embedding)

#query="what is Cybersecurity"
def top_k_similar(query,top_k=3):

    query_embedding=create_query_embedding(query)
    chunks=create_chunks("document.txt")
    doc_embedding=create_doc_embedding(chunks)
    similarities= cosine_similarity(doc_embedding,[query_embedding])
    #since query embed is 1d and doc embed is 2d


    similarities=similarities.flatten()
    #argsort()-return indexex according to value form small to high
    #-top[k:]-return last k 
    #[::-1]- make it  reverse so that highest lies in first
    top_k_indexes=similarities.argsort()[-top_k:][::-1]
    results=[]
    for index in top_k_indexes:
        results.append({
            "chunk":chunks[index],
            "score":similarities[index]
        })
    return results
