
from langchain_community.embeddings import OllamaEmbeddings
import tiktoken
import numpy as np



# Indexing:

question = 'what kind of pets do i like ?'
document = 'My favorite pet is a cat.'

def num_token_from_string(srting, encoding_name):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(srting))
    return num_tokens



num_token_from_string(question, 'cl100k_base')
print(x)


embd = OllamaEmbeddings(model="gemma:2b")
query_result = embd.embed_query(question)
document_result = embd.embed_query(document)

x = len(query_result)
print(x)



def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

similarity = cosine_similarity(query_result, document_result)
print(similarity)
