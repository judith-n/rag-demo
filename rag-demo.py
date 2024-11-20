# %% [markdown]
# Load data and split into chunks

# %%
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_folder_path = "data_files"
documents_list = []
for file in os.listdir(pdf_folder_path):
    if file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder_path, file)
        loader = PyMuPDFLoader(pdf_path)
        documents_list.extend(loader.load())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
documents = text_splitter.split_documents(documents_list)

# %%
print("Number of chunks created: ", len(documents))

# %% [markdown]
# Create embeddings and store them in vector database

# %%
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Instantiate the embedding model
embedder = OllamaEmbeddings(
    model="nomic-embed-text"
)
# Create the vector store 
vector = FAISS.from_documents(documents, embedder)

# %% [markdown]
# Retrieve data

# %%
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
retrieved_docs = retriever.invoke("How do I order a multisport card for my child?")
print(retrieved_docs)

# %% [markdown]
# Setup LLM model

# %%
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="gemma2:2b")

# %% [markdown]
# Generate response using LLM

# %%
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.globals import set_verbose
from langchain.globals import set_debug

#uncomment to see prompt info 
#set_debug(True)
prompt = hub.pull("rlm/rag-prompt")

# human

# [INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<</SYS>> 

# Question: {question} 

# Context: {context} 

# Answer: [/INST]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What to do if I lost my multisport card?")


# %%
rag_chain.invoke("What is the number for a dedicated Medicover contact?")

# %%
rag_chain.invoke("How do I add my family to Uniqua insurance?")

# %%
rag_chain.invoke("How do I add my partner to Uniqua insurance?")

# %%
rag_chain.invoke("How long should I wait for a refund from Medicover?")


