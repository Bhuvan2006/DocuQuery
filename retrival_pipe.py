from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv 
from ingestion_pipe import GeminiEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage



load_dotenv()


chat_history = []

# embedding model
# model1= GeminiEmbeddings(model="gemini-embedding-001")

# changeing rag


# Connect to your document database
persistent_directory = "db/chroma_db"
embeddings = GeminiEmbeddings(model="gemini-embedding-001")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Set up AI model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    # Step 1: Make the question clear using conversation history
    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ]+chat_history+[
            HumanMessage(content=user_question)
        ]
        
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Step 2: Retrieve relevant documents from the database    
    retriver = db.as_retriever(search_kwargs={"k": 5})
    docs = retriver.invoke(search_question)
    
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")
    
    # Step 3: Create final prompt
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """
    
    # Step 4: Get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = result.content
    
    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"Answer: {answer}")
    return answer

# Simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()
    





















'''
# Basic RAG

# retrieval data from database
persist_directory="db/chroma_db"

db= Chroma(
    persist_directory=persist_directory,
    embedding_function=model,
    collection_metadata={"hnsw:space":"cosine"}
)

# search for 5 docs close to query
retriever= db.as_retriever(search_kwargs={"k":5})

# user query
query="how we are plotting those local minimum"
relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")

# Display results

print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


#model
# model1= ChatGoogleGenerativeAI(model="gemini-1.0-pro")

# LLM query preparation    --- char(10)= '\n'---
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents and online refrences on it. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

#generate response
from langchain_google_genai import ChatGoogleGenerativeAI


result = llm.invoke(messages)




# Displaying response
print("\n --- Response --- \n")

# print("\nFull result:")
# print(result)

print("\nContent only:")
print(result.content) 

'''