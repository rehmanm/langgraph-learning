from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from typing import List, TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import  HumanMessage, SystemMessage, get_buffer_string, AIMessage
from pydantic import BaseModel, Field

class TravelDestination(BaseModel):
    city: str = Field(
        description = "City of the travel destination"
    )
    country: str = Field(
        description = "Country of the travel destination"
    )
    popular_places: list[str] = Field(
        description = "Popular places to visit in the travel destination"
    )
    startDate: str = Field(
        description = "Start date of the travel"
    )
    endDate: str = Field(
        description = "End date of the travel"
    )
    recommended_star: int = Field(
        description = "Recommended star rating for the destination (1-5)"
    )
    estimated_budget: int = Field(
        description = "Estimated budget for the destination"
    )

class TravelDestinationList(BaseModel):
    travelDestinations: list[TravelDestination] = Field(
        description = "Comprehensive list of travel destinations with their details"
    )

class TravelAssistantState(TypedDict):
    budget: float
    max_destinations: int
    preferences: list[str]
    human_feedback: str
    travel_recommendations: List[TravelDestination] 
  
      

def generate_travel_recommendations(state: TravelAssistantState):
    """Generate travel recommendations based on user preferences"""
    max_destinations = state["max_destinations"]
    preferences = ", ".join(state["preferences"])
    human_feedback = state.get("human_feedback", "")
    travel_instructions = f""" 

    1. You are a travel assistant. Based on the user's preferences: {preferences}, suggest up to {max_destinations} travel destinations.
    For each destination, provide the city name, country, popular places to visit, start date, end date and recommended star rating.

    2. Suggest the best time to visit each destination based on weather and local events.

    3. Don't suggest one country multiple times. Each destination should be unique.

    4. Examine the user's feedback after providing the recommendations. If the feedback is 'approve', conclude the process.
    If the feedback is anything other than 'approve', return to step 1 and generate a new set of recommendations.
    {human_feedback}

    5. Give star ratings (1-5) for each destination based on overall appeal and user preferences.  

    6. Consider budget constraints while suggesting destinations.

    7. Don't suggest more than {max_destinations} destinations.
    """
    
    structured_llm = model.with_structured_output(TravelDestinationList)

    system_message = travel_instructions.format(
        preferences=preferences,
        max_destinations=max_destinations,
        human_feedback=human_feedback
    )

    travel_destinations = structured_llm.invoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(content="Please provide your travel recommendations.")
        ]
    )
    
    return {
        "travel_recommendations": travel_destinations.travelDestinations
    }

def human_feedback(state: TravelAssistantState):
    """No-op node that should be interupted on"""
    pass

from langgraph.types import Send
def initiate_travel_itineraries(state: TravelAssistantState):
    human_feedback=state.get('human_feedback','approve')
    if human_feedback.lower() != 'approve':
        # Return to create_analysts
        return "generate_recommendations"
    
    return [Send("generate_travel_itinerary", {
                    'travel_destination': travel_destination
            })
                 for travel_destination in state['travel_recommendations']
                  ]


import operator

class TravelItineraryState(TypedDict):
    travel_destination: TravelDestination
    travel_itineraries: Annotated[list, operator.add]
    best_travel_destination: str


def generate_travel_itinerary(state: TravelItineraryState):
    """Generate a detailed travel itinerary for a given travel destination"""
    iterary_instructions = """
    You are a travel assistant. Based on the travel destination details provided, create a detailed travel itinerary.
    Include daily activities, places to visit, dining options, and any special events happening during the travel dates.
    Consider the recommended star rating and budget while planning the itinerary for the destination
    {city}, {country} from {startDate} to {endDate} with a 
    recommended star rating of {recommended_star} 
    and an estimated budget of {estimated_budget}.

    1. Share the itinerary in a day-wise format, ensuring a balance between sightseeing, relaxation, and local experiences.
    2. Share any travel tips or important information relevant to the destination.
    3. Share the travel plan including accommodation, transportation, and activities.
    4. Find the booking links for popular places to visit.
    5. Return the itinerary in the below format:
    Itinerary for {city}, {country}:
    Day 1:
    - Activity 1
    - Activity 2
    Day 2:
    - Activity 1
    - Activity 2
    """
     
    travel_destination = state["travel_destination"]

    system_message = iterary_instructions.format(
        city =  travel_destination.city,
        country = travel_destination.country,
        startDate = travel_destination.startDate,
        endDate = travel_destination.endDate,
        recommended_star = travel_destination.recommended_star,
        estimated_budget = travel_destination.estimated_budget
    )

    travel_itinerary = model.invoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(content="Please generate a detailed travel itinerary including daily activities, places to visit, dining options, and any special events.")
        ]
    )
    
    return {
        "travel_itineraries": [travel_itinerary.content]
    }

def recommend_best_destination(state: TravelItineraryState):
    """Suggest the best travel destination from the recommendations"""
    travel_itineraries = state["travel_itineraries"]
    
    recommendation_instructions = """ 

    You are a travel assistant. Based on the travel itineraries provided, suggest the best travel destination.

    {context}

    1. Evaluate each travel itinerary based on the activities, places to visit, dining options, and special events.
    2. Consider the overall appeal, user preferences, and budget constraints while making the recommendation.
    3. Return the best travel destination along with a brief explanation of why it stands out among the others. 
    4. Return the travel itinerary as well for the best travel destination.
    """
    context = "\n\n".join(travel_itineraries)

    system_message = recommendation_instructions.format(
        context = context   
    )

    best_recommendation = model.invoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(content="Please suggest the best travel destination based on the provided itineraries.")
        ]
    )
    
    return {
        "best_travel_destination": best_recommendation.content
    }
 

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
travel_assistant_builder = StateGraph(TravelAssistantState)
travel_assistant_builder.add_node("generate_recommendations", generate_travel_recommendations)
travel_assistant_builder.add_node("human_feedback", human_feedback)
travel_assistant_builder.add_node("generate_travel_itinerary", generate_travel_itinerary)
travel_assistant_builder.add_node("recommend_best_destination", recommend_best_destination)

travel_assistant_builder.add_edge(START, "generate_recommendations")
travel_assistant_builder.add_edge("generate_recommendations", "human_feedback")
travel_assistant_builder.add_conditional_edges("human_feedback", initiate_travel_itineraries, ["generate_recommendations", 'generate_travel_itinerary'])
travel_assistant_builder.add_edge("generate_travel_itinerary", "recommend_best_destination")
travel_assistant_builder.add_edge("recommend_best_destination", END)

memory = MemorySaver()

from langgraph.checkpoint.memory import MemorySaver
graph = travel_assistant_builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)

