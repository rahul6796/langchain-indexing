
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import WebBaseLoader 
from langchain_community.vectorstores import Chroma

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate

import bs4




# # load documents:
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent",),
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

# reteriver = vectorstore.as_retriever()


# Retriveral:

reteriver  = vectorstore.as_retriever(search_kwargs = {'k':1})
docs = reteriver.get_relevant_documents('What is Task Decompostion ?')


# Generations

template = """Answer the question based only on the following 
context: {context}

Question: {question}

"""

prompt= ChatPromptTemplate.from_template(template)


# LLM 
llm = Ollama(model = 'gemma:2b')

# chain = 
chain = prompt | llm

response = chain.invoke({'context':docs, 'question':"What is Task Decompostion ?"})
print(response)
