from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0) 


def add(x: float, y:float) -> float:
    """Add 'x' and 'y'."""
    return x + y

 
def subtract(x: float, y:float) -> float:
    """Subtract 'x' and 'y'."""
    return x - y


def multiply(x: float, y:float) -> float:
    """Multiply 'x' and 'y'."""
    return x * y


def divide(x: float, y:float) -> float:
    """Divide 'x' and 'y'."""
    return x / y


def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the power of 'y'."""
    return x ** y

tools = [add, subtract, multiply, divide, exponentiate]

llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="You are a helpful assistant tasked with writing performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile graph
graph = builder.compile()