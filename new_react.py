import json
from typing import TypedDict,Annotated,Sequence
from langchain_core.messages import HumanMessage,ToolMessage,BaseMessage,SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from dotenv import load_dotenv # used to store secret stuff like AF
import os
from langgraph.prebuilt import ToolNode
from IPython.display import Image
from langchain_groq import ChatGroq
import requests
from serpapi import GoogleSearch
load_dotenv()
api_key = os.getenv("XAI_API_KEY")
amadeus_key = os.getenv("AMADEUS_API_KEY")
amadeus_secret = os.getenv("AMADEUS_API_SECRET")
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")
if not api_key:
    raise ValueError("🚨 GROQ_API_KEY not found. Check .env file!")

if not amadeus_key or not amadeus_secret:
    raise ValueError("🚨 Amadeus credentials missing")

if not SERPAPI_KEY:
    raise ValueError("❌ SERPAPI_API_KEY missing")

class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]

model = ChatGroq(
    model="openai/gpt-oss-120b",  # Groq-supported OSS model
    api_key=api_key,
    temperature=0,
)

def get_amadeus_token():
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": amadeus_key,
        "client_secret": amadeus_secret
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post(url, data=payload, headers=headers)
    return r.json()["access_token"]


@tool
def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    adults: int = 1
):
    """
    Search flights from origin to destination on a given date.
    """
    token = get_amadeus_token()

    url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    params = {
        "originLocationCode": origin,
        "destinationLocationCode": destination,
        "departureDate": departure_date,
        "adults": adults,
        "currencyCode": "EUR",
        "max": 5
    }

    response = requests.get(url, headers=headers, params=params)
    return response.json()


@tool
def search_hotels(
    location: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 1,
):
    """
    Search hotels and prices using Google Hotels (SerpApi).

    - location: City or place name (e.g. "Bali", "Mumbai")
    - Dates must be YYYY-MM-DD
    """

    params = {
        "engine": "google_hotels",
        "q": location,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": adults,
        "currency": "USD",
        "api_key": SERPAPI_KEY,
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    properties = results.get("properties")
    if not properties:
        raise RuntimeError("No hotel data returned from SerpApi")

    hotels = []
    for prop in properties[:5]:
        hotels.append(
            {
                "name": prop.get("name"),
                "price_per_night": prop.get("rate_per_night", {}).get("lowest"),
                "rating": prop.get("overall_rating"),
                "reviews": prop.get("reviews"),
                "link": prop.get("link"),
            }
        )

    return {
        "location": location,
        "check_in": check_in_date,
        "check_out": check_out_date,
        "hotels": hotels,
    }


tools=[search_flights,search_hotels]
model_with_tools = model.bind_tools(tools)
def model_call(state: AgentState) -> AgentState:
    """
    Coordinator Agent:
    - Understands user intent
    - Routes to Flight Agent, Hotel Agent, or both
    - Maintains context across turns
    """

    system_prompt = SystemMessage(
        content="""
You are a travel coordinator AI.

====================
HOTEL SEARCH RULES
====================
- Use search_hotels for hotels.
- Hotels are searched by LOCATION NAME (not city codes).
- Always call search_hotels when location + dates are provided.
- Do NOT ask for hotel IDs.
- Do NOT invent prices.

Hotel tool:
search_hotels(location, check_in_date, check_out_date, adults)

====================
FLIGHT SEARCH RULES
====================
- Flights require origin, destination, and departure_date.
- Use IATA city/airport codes internally.

====================
GENERAL RULES
====================
- Dates must be YYYY-MM-DD.
- Ask follow-ups ONLY if required fields are missing.
- After tools run, summarize clearly.
"""
    )


    response = model_with_tools.invoke(
        [system_prompt] + state["messages"]
    )

    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    # Iterate backwards to find the last AI message that has tool_calls attribute
    for msg in reversed(messages):
        if hasattr(msg, "tool_calls"):
            if not msg.tool_calls:
                return "end"
            else:
                return "continue"
    # If no AI message found with tool_calls, end the flow
    return "end"


graph = StateGraph(AgentState)
graph.add_node("our_agent",model_call) 
graph.add_edge (START, "our_agent") 
tool_node = ToolNode(tools=tools)
graph.add_node("tools",tool_node)
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue":"tools",
        "end":END
    }
)
graph.add_edge("tools","our_agent")
agent = graph.compile()

# png_bytes = agent.get_graph().draw_mermaid_png()  # no arguments
# # If in a notebook, to display:
# Image(png_bytes)

# # To save to a file:
# with open("graph.png", "wb") as f:
#     f.write(png_bytes)

# print("Graph saved as graph.png")
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        message.pretty_print()


# print("\n🧭 Travel Assistant Started (type 'exit' to quit)\n")
conversation_history = []

# while True:
#     user_query = input("🧭 Enter your travel query: ").strip()

#     # Exit condition
#     if user_query.lower() == "exit":
#         print("\n👋 Exiting Travel Assistant. Goodbye!\n")
#         break

#     inputs = {
#         "messages": [
#             HumanMessage(content=user_query)
#         ]
#     }

#     print_stream(
#         agent.stream(inputs, stream_mode="values")
#     )
while True:
    user_query = input("🧭 Enter your travel query: ").strip()

    if user_query.lower() == "exit":
        break

    conversation_history.append(HumanMessage(content=user_query))

    result = agent.invoke({"messages": conversation_history})

    final_message = result["messages"][-1]
    final_message.pretty_print()

    conversation_history.append(final_message)








