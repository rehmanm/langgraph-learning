
from langgraph.graph import StateGraph, START, END

from langgraph.checkpoint.memory import MemorySaver
from typing import List, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.messages import  HumanMessage, SystemMessage
from pydantic import BaseModel, Field

class Analyst(BaseModel):
    affiliation: str = Field(
        description = "Primary affliation of the analyst"
    )
    name: str = Field(
        description = "Full name of the analyst"
    )
    role: str = Field(
        description = "Role of the analyst in the context of the topic"
    )
    description: str = Field(
        description = "Description of the analyst focus, concerns, and motives"
    )
    @property
    def persona(self) -> str:
        return f"{self.name}\nRole:{self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}."
    
class Perspectives(BaseModel):
    analysts: list[Analyst] = Field(
        description = "Comprehensive list of analysts with their roles and affliations"
    )

class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analysts_feedback: str
    analysts: List[Analyst]

analyst_instructions = """
You are tasked with creating a set of AI analysts personas. Follow the instructions carefully:
1. First review the research topic:
{topic}

2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts:

{human_analysts_feedback}

3. Determine the most interesting themes based upton the documents and / or feedback above.

4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme.
"""
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

def create_analysts(state: GenerateAnalystsState):

    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_analysts_feedback = state.get("human_analysts_feedback", "")

    structured_llm = model.with_structured_output(Perspectives)

    system_message = analyst_instructions.format(
        topic=topic,
        max_analysts=max_analysts,
        human_analysts_feedback=human_analysts_feedback
    )

    analysts = structured_llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Generate the set of analysts. ")])
    return {
        "analysts": analysts.analysts
    }


def human_feedback(state: GenerateAnalystsState):
    """No-op node that should be interupted on"""
    pass

def should_continue(state: GenerateAnalystsState):
    """Return the next node to execute"""
    human_feedback = state.get("human_analysts_feedback", None)
    if human_feedback:
        return "create_analysts"
    
    return END

analyst_builder = StateGraph(GenerateAnalystsState)
analyst_builder.add_node("create_analysts", create_analysts)
analyst_builder.add_node("human_feedback", human_feedback)
analyst_builder.add_edge(START, "create_analysts")
analyst_builder.add_edge("create_analysts", "human_feedback")
analyst_builder.add_conditional_edges("human_feedback", should_continue, ["create_analysts", END])

memory = MemorySaver()

analyst_graph = analyst_builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)

graph = analyst_graph



