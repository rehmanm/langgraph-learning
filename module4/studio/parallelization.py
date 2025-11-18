import operator
from typing import Annotated
from typing_extensions import TypedDict

from langchain_tavily import TavilySearch
from langchain_community.document_loaders.wikipedia import WikipediaLoader

from langchain_core.messages  import HumanMessage, SystemMessage
from langchain_core.documents import Document

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0) 


class State(TypedDict):
    question: str
    answer: str
    context: Annotated[list, operator.add]


def search_web(state):

    """"Rereive docs from the web search"""

    # Search
    tavily_search = TavilySearch(max_results=3)
    data = tavily_search.invoke({"query": state["question"]})
    search_docs = data.get("results", data)

    # Format

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{ doc["url"] }">\n{doc["content"]} \n</Document>'
            for doc in search_docs
        ]
    )

    return {
        "context": [formatted_search_docs]
    }

def search_wikipedia(state):

    """"Rereive docs from Wikipedia"""

     # Search

    search_docs = WikipediaLoader(query=state["question"],
                                   load_max_docs = 2).load()
    
    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{ doc.metadata["source"] } page="{doc.metadata.get("page", "")}"">\n{doc.page_content} \n</Document>'
            for doc in search_docs
        ]
    )

    return {
        "context": [formatted_search_docs]
    }

def generate_answer(state):

    """Generate answer based on retrieved docs"""

    # Get State
    context = state["context"]
    question = state["question"]

    # Template
    answer_template = """Answer the question {question} using this context: {context}"""
    answer_instruction = answer_template.format(
        question=question,
        context=context
    )

    # Answer
    answer = llm.invoke(
        [
            SystemMessage(content=answer_instruction),
            HumanMessage(content="Answer the question")
        ]
    )

    return {
        "answer": answer
    }


builder = StateGraph(State)

builder.add_node(
    "search_web", search_web
    )
builder.add_node(
    "search_wikipedia", search_wikipedia
)
builder.add_node(
    "generate_answer", generate_answer
)


builder.add_edge(START, "search_wikipedia")
builder.add_edge("search_wikipedia", "generate_answer")
builder.add_edge(START, "search_web")
builder.add_edge("search_web", "generate_answer")
builder.add_edge("generate_answer", END)

graph = builder.compile()