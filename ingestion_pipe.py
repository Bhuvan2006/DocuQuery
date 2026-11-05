import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

import os

from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings

load_dotenv()


client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

class GeminiEmbeddings(Embeddings):
    def __init__(self, model="gemini-embedding-001", output_dimensionality=768):
        self.client = genai.Client()
        self.model = model
        self.output_dimensionality = output_dimensionality

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            result = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=self.output_dimensionality)
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings

    def embed_query(self, text):
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=self.output_dimensionality)
        )
        return result.embeddings[0].values





def load_doc(docs_path="docs"):
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"File not found at {docs_path}")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )
    
    documents = loader.load()
    
    if len(documents)==0:
        raise FileNotFoundError(f"file not found {docs_path}")
    
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")
    return documents
    
    
def split_documents(documents, chunk_size=800,chunk_overlap=0):
    print("Inside performing chunks \n")
    
    text = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks= text.split_documents(documents)
    
    if chunks:
        
        for i,chunk in enumerate(chunks[:3]):
            print(f"\n chunk {i+1}\n")
            print(f"\n Metadata: {chunk.metadata['source']}\n")
            print(f"\n length: {len(chunk.page_content)} characters\n")
            print(f"\n Content:")
            print(f"\n {chunk.page_content}\n")
            print("-"*20)
            
    return chunks
            
         
def vector_store(chunks,persist_directory="db/chroma_db"):
    
    vectorStore= Chroma.from_documents(
        documents=chunks,
        embedding= GeminiEmbeddings(model="gemini-embedding-001", output_dimensionality=768),
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"} # hosw
    )
    
    print("  --created vectorstore-- \n")
    
    print(f"vector store saved to {persist_directory}")
    
    return vectorStore
    
       
def main():
    print("hello in main")
    
    documents = load_doc("docs")
    
    chunks= split_documents(documents)
    
    vectordb= vector_store(chunks)
    

    
if __name__ == "__main__":
    # print("API Key:", os.getenv("OPENAI_API_KEY"))
    main()