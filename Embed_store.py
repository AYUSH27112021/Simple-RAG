from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from Data_preprocessing import pre_processed_data
import tempfile
import os

def get_retriever():
    persist_path = os.path.join(tempfile.gettempdir(), "union.parquet")
    doc_splits=pre_processed_data()

    # Add to vectorDB
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
        persist_path=persist_path,
        serializer="parquet",
    )

    # Create retriever
    retriever = vectorstore.as_retriever(k=3)
    return retriever