from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import WebBaseLoader 
from langchain_community.vectorstores import Chroma

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from operator import itemgetter


import bs4




# # load documents:
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs = dict(
        parse_only = bs4.SoupStrainer(
            class_ = ('post-content', 'post-title', 'post-header')
        )
    ),

)

docs = loader.load()


# # splitter 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap=300, 
)

splits = text_splitter.split_documents(docs)


# # Vector dataBase Chroma:

emb = OllamaEmbeddings(model = 'gemma:2b')
vectorstore = Chroma.from_documents(documents = splits, embedding = emb)
reteriver  = vectorstore.as_retriever()



llm = Ollama(model = 'gemma:2b')
# Prompt


# multi-query with different purspective

template = """You are an AI language model assistant. Your Task is to generate five different version of the given user question to retrieve
documents from a vector database. By generating multiple perspective on the user question, your goal is to help 
the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions seperated by newline. Original question: {question}"""

prompt_perspective = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_perspective
    | llm
    | StrOutputParser()
    | (lambda x: x.split('\n'))
)




def get_unique_union(documents):

    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))

    return [loads(doc) for doc in unique_docs]

question  = "What is task decomposition for LLM agents?"
retrieval_chain = generate_queries |reteriver.map() |get_unique_union

docs = retrieval_chain.invoke({'question': question})




# RAG:
template = """Answer the following question based on this context:
{context}

Question: {question}"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    {'context': retrieval_chain,
     'question': itemgetter('question')}
    | prompt
    | llm
    | StrOutputParser()
)

response = final_rag_chain.invoke({'question': question})
print(response)