from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS 
from langgraph.graph import StateGraph


from dotenv import load_dotenv

def setup_env():
    """Load environment variables from .env file."""
    load_dotenv()

def create_llm() -> ChatGoogleGenerativeAI:
    """Create and return the LLM instance."""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.0)
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        raise

def create_vectorstore(docs: list[str]) -> FAISS:
    """Create and return a FAISS vectorstore from documents."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_texts(docs, embedding=embeddings)

setup_env()
llm = create_llm()



# Step 2: Create a small document store
docs = [
    "Cats are small, furry animals that like to sleep a lot.",
    "Dogs are loyal and love to play fetch.",
    "Birds can fly and often sing in the morning."
]

def get_docs() -> list[str]:
    """Return a list of sample documents."""
    return [
        "Cats are small, furry animals that like to sleep a lot.",
        "Dogs are loyal and love to play fetch.",
        "Birds can fly and often sing in the morning."
    ]

docs = get_docs()
vectorstore = create_vectorstore(docs)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore =  FAISS.from_texts(docs, embedding=embeddings)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=vectorstore.as_retriever()
)

def answer_question(state):
    question = state["question"]
    answer_dict = qa_chain.invoke({"query": question})
    print(f"answer_dict: {answer_dict}")
    return {
        "answer": answer_dict["result"],
        "question": question
    }

# Step 3: Create a StateGraph and add the node

from typing import TypedDict

class RAGState(TypedDict):
    question: str
    answer: str
 

graph = StateGraph(RAGState)
graph.add_node("RAG", answer_question)
graph.set_entry_point("RAG")

app = graph.compile()

question = "Whose more loyal"

result = app.invoke({"question": question})   
print("--- RAG Demo ---")
print(f"Question: {question}")
print(f"Answer: {result}")