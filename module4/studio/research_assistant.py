import operator
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from typing import List, TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.messages import  HumanMessage, SystemMessage, get_buffer_string, AIMessage
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WikipediaLoader


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

class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list

class SearchQuery(BaseModel):
    search_query: str = Field(
        description="search query for retrieval"
    )

question_instructions = """You are an analyst tasked with interviewing an expert to learn about a specific topic. 

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.
        
2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {goals}
        
Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.
        
When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""

def generate_questions(state: InterviewState):

    analyst = state["analyst"]
    messages = state["messages"]
    system_message = question_instructions.format(
        goals=analyst.persona
    )
    question = model.invoke([SystemMessage(content=system_message)] + messages)

    return{
        "messages": [question]
    }

tavily_search = TavilySearch(max_results=3, verify_ssl=False)

# Search query writing
search_instructions = SystemMessage(content=f"""You will be given a conversation between an analyst and an expert. 

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
        
First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query""")

def search_web(state: InterviewState):

    structured_llm = model.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke(
        [search_instructions] + state["messages"]
        )
    
    data = tavily_search.invoke({
        "query": search_query.search_query
    })

    search_docs = data.get("results", data)
    print(f"search_docs: {search_docs}")

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )
    return{
        "context": [formatted_search_docs]
    }

def search_wikipedia(state: InterviewState):
    structured_llm = model.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke(
        [search_instructions] + state["messages"]
        )
    
    search_docs = WikipediaLoader(
        query=search_query.search_query,
        load_max_docs=2
    ).load()

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


answer_instructions = """You are an expert being interviewed by an analyst.

Here is analyst area of focus: {goals}. 
        
You goal is to answer a question posed by the interviewer.

To answer question, use this context:
        
{context}

When answering questions, follow these guidelines:
        
1. Use only the information provided in the context. 
        
2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

3. The context contain sources at the topic of each individual document.

4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. 

5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc
        
6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list: 
        
[1] assistant/docs/llama3_1.pdf, page 7 
        
And skip the addition of the brackets as well as the Document source preamble in your citation."""

def generate_answer(state: InterviewState):
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    system_message = answer_instructions.format(
        goals=analyst.persona,
        context=context
    )
    answer = model.invoke([SystemMessage(content=system_message)] + messages)
    answer.name = "expert"
    return{
        "messages": [answer]
    }

def save_inteview(state: InterviewState):
    messages = state["messages"]
    interview = get_buffer_string(messages)
    return{
        "interview": interview
    }

def route_messages(state: InterviewState, 
                  name: str = "expert"):
    messages = state["messages"]
    max_num_turns  = state.get("max_num_turns", 2)

    num_responses = len(
        [msg for msg in messages if isinstance(msg, AIMessage) and msg.name == name]
        )
    
    if num_responses >= max_num_turns:
        return "save_interview"
    
    last_question = messages[-2]

    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return "ask_question"

section_writer_instructions = """You are an expert technical writer. 
            
Your task is to create a short, easily digestible section of a report based on a set of source documents.

1. Analyze the content of the source documents: 
- The name of each source document is at the start of the document, with the <Document tag.
        
2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers
        
3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

4. Make your title engaging based upon the focus area of the analyst: 
{focus}

5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Create a numbered list of source documents, as you use them
- Do not mention the names of interviewers or experts
- Aim for approximately 400 words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
        
6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

7. Be sure to combine sources. For example this is not correct:

[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

There should be no redundant sources. It should simply be:

[3] https://ai.meta.com/blog/meta-llama-3-1/
        
8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""

def write_section(state: InterviewState):
    interview = state["interview"]
    analyst = state["analyst"]
    context = state["context"]

    system_message = section_writer_instructions.format(
        focus = analyst.description
    )

    section = model.invoke(
        [
            SystemMessage(content=system_message) 
        ] 
        +
        [
            [HumanMessage(content=f"Use this source to write your section: {context}")]           
        ]
    )

    return{
        "sections": section.content
    }

interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_questions", generate_questions)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wikipedia", search_wikipedia)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_inteview)
interview_builder.add_node("write_section", write_section)

interview_builder.add_edge(START, "ask_questions")
interview_builder.add_edge("ask_questions", "search_web")
interview_builder.add_edge("ask_questions", "search_wikipedia")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_conditional_edges("answer_question", route_messages, ["ask_questions", "save_interview"])
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

interview_graph = interview_builder.compile(checkpointer=memory).with_config(run_name="Conduct Interviews")


graph = analyst_graph



