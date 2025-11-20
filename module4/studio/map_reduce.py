
from typing import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
import operator
from pydantic import BaseModel
from langgraph.constants import Send

subjects_prompt = """Generate a comma sperated list of between 2 and 5 examples related to {topic}."""
joke_prompt = """Generate a funny joke about {subject}."""
best_joke_prompt = """Below are the bunch of jokes about {topic}. Select the best one! and return the ID of the best one. starting 0 as the id of the first joke. Jokes: \n\n  {jokes}"""


model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)

class Subjects(BaseModel):
    subjects: list[str]

class BestJoke(BaseModel):
    id: int

class OverAllState(TypedDict):
    topic: str
    subjects: list
    jokes: Annotated[list, operator.add]
    best_selected_joke: str

def generate_topics(state: OverAllState):
    """Generate subjects related to the topic"""

    prompt = subjects_prompt.format(topic=state["topic"])
    response = model.with_structured_output(Subjects).invoke(prompt) 
    return {
        "subjects": response.subjects
    }

def continue_to_jokes(state: OverAllState):
    """Generate jokes for each subject"""

    subjects = state["subjects"]
    return [Send("generate_joke", {"subject": subject}) for subject in subjects]


class JokeState(TypedDict):
    subject: str

class Joke(BaseModel):
    joke: str

def generate_joke(state: JokeState):
    """Generate a joke about the subject"""

    prompt = joke_prompt.format(subject=state["subject"])
    response = model.with_structured_output(Joke).invoke(prompt)
    return {
        "jokes": [response.joke]
    }


def best_joke(state: OverAllState):
    """Select the best joke from the list"""

    jokes = "\n\n".join(state["jokes"])
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
    response = model.with_structured_output(BestJoke).invoke(
        prompt
    )
    best_joke = state["jokes"][response.id]
    return {
        "best_selected_joke": best_joke
    }

builder = StateGraph(OverAllState)
builder.add_node("generate_topics", generate_topics)
builder.add_node("generate_joke", generate_joke)
builder.add_node("best_joke", best_joke)

builder.add_edge(START, "generate_topics")
builder.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
builder.add_edge("generate_joke", "best_joke")
builder.add_edge("best_joke", END)

graph = builder.compile()
