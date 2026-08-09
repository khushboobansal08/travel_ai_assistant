import json
from typing import TypedDict,Annotated,Sequence
from langchain_core.messages import HumanMessage,ToolMessage,BaseMessage,SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from dotenv import load_dotenv # used to store secret stuff like AF
import os
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import requests
from serpapi import GoogleSearch
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
load_dotenv()
# api_key = os.getenv("XAI_API_KEY")
amadeus_key = os.getenv("AMADEUS_API_KEY")
amadeus_secret = os.getenv("AMADEUS_API_SECRET")
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")
openrouter_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_key:
    raise ValueError("🚨 open router key not found. Check .env file!")

if not amadeus_key or not amadeus_secret:
    raise ValueError("🚨 Amadeus credentials missing")

if not SERPAPI_KEY:
    raise ValueError("❌ SERPAPI_API_KEY missing")

class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]

# model = ChatGroq(
#     model="openai/gpt-oss-120b",  # Groq-supported OSS model
#     api_key=api_key,
#     temperature=0,
# )
model = ChatOpenAI(
    model="openrouter/auto:free",  # ⚡ FAST + unlimited
    # Alternative: "microsoft/phi-3-mini-128k-instruct:free" 
    api_key=openrouter_key,
    base_url="https://openrouter.ai/api/v1",
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
    stay_length: int,
    adults: int = 1,
):
    """
    Search hotels and prices using Google Hotels (SerpApi).

    - location: City or place name (e.g. "Bali", "Mumbai")
    - Dates must be YYYY-MM-DD
    """
    from datetime import datetime, timedelta
    check_out_date = (datetime.fromisoformat(check_in_date) + timedelta(days=stay_length)).isoformat()
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

class PrettyToolNode:
    """Formats tool results for clean user output"""
    def __init__(self, tools):
        self.tools = {tool.name: tool for tool in tools}

    def __call__(self, state):
        messages = state['messages']
        last_message = messages[-1]
        
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return state
        
        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            # Execute tool
            result = self.tools[tool_name].invoke(tool_args)
            
            # FORMAT → Clean tables only
            if tool_name == 'search_flights':
                formatted = self._format_flights(result)
            elif tool_name == 'search_hotels':
                formatted = self._format_hotels(result)
            else:
                formatted = result
            
            tool_msg = ToolMessage(
                content=formatted,
                tool_call_id=tool_call['id']
            )
            tool_messages.append(tool_msg)
        
        return {'messages': tool_messages}

    def _format_flights(self, data):
        """✅ PROPER HTML TABLE for flights"""
        flights = data.get('data', [])
        if not flights:
            return "❌ No flights found"
        
        html = """
        <div style="font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; margin: 10px 0;">
        <h3 style="color: #1E88E5; margin: 0 0 15px 0;">✈️ Flights Found</h3>
        <table style="border-collapse: collapse; width: 100%; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <thead>
                <tr style="background: linear-gradient(135deg, #1E88E5 0%, #1976D2 100%); color: white;">
                    <th style="padding: 15px 12px; text-align: left; font-weight: 600;">Flight</th>
                    <th style="padding: 15px 12px; text-align: left; font-weight: 600;">Timing</th>
                    <th style="padding: 15px 12px; text-align: right; font-weight: 600;">Price</th>
                    <th style="padding: 15px 12px; text-align: right; font-weight: 600;">Seats</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for i, flight in enumerate(flights[:3], 1):
            seg = flight['itineraries'][0]['segments'][0]
            flight_num = f"{seg['carrierCode']}{seg['number']}"
            depart = seg['departure']['at'][:16].replace('T', ' ')
            arrive = seg['arrival']['at'][:16].replace('T', ' ')
            duration = flight['itineraries'][0]['duration']
            price = flight['price']['total']
            seats = flight['numberOfBookableSeats']
            
            html += f"""
            <tr style="border-bottom: 1px solid #E0E0E0;">
                <td style="padding: 15px 12px; font-weight: 500;">{i}. <strong>{flight_num}</strong></td>
                <td style="padding: 15px 12px;">{depart} → {arrive}<br><small style="color: #666;">({duration})</small></td>
                <td style="padding: 15px 12px; text-align: right; color: #2E7D32; font-weight: 600; font-size: 16px;">€{price}</td>
                <td style="padding: 15px 12px; text-align: right; color: #388E3C;">{seats} <small>seats</small></td>
            </tr>
            """
        
        html += """
            </tbody>
        </table>
        </div>
        """
        return html

    def _format_hotels(self, data):
        """✅ FIXED: Handle float ratings safely"""
        hotels = data.get('hotels', [])
        if not hotels:
            return "❌ No hotels found"
        
        html = """
        <div style="font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; margin: 10px 0;">
        <h3 style="color: #FB8C00; margin: 0 0 15px 0;">🏨 Top Hotels</h3>
        <table style="border-collapse: collapse; width: 100%; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <thead>
                <tr style="background: linear-gradient(135deg, #FB8C00 0%, #F57C00 100%); color: white;">
                    <th style="padding: 15px 12px; text-align: left; font-weight: 600;">Hotel</th>
                    <th style="padding: 15px 12px; text-align: right; font-weight: 600;">Price/Night</th>
                    <th style="padding: 15px 12px; text-align: right; font-weight: 600;">Rating</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for hotel in hotels[:3]:
            name = hotel.get('name', 'N/A')
            price = hotel.get('price_per_night', 'N/A')
            rating = hotel.get('rating')  # Could be float, str, None
            
            # ✅ FIXED: Safe rating handling
            rating_display = rating if rating else 'N/A'
            stars = '★' * int(float(rating) // 1) if isinstance(rating, (int, float)) and rating else ''
            
            html += f"""
            <tr style="border-bottom: 1px solid #E0E0E0;">
                <td style="padding: 15px 12px; font-weight: 500;"><strong>{name}</strong></td>
                <td style="padding: 15px 12px; text-align: right; color: #2E7D32; font-weight: 600;">${price}</td>
                <td style="padding: 15px 12px; text-align: right;">
                    <span style="color: #FFB300;">{stars}</span> {rating_display}
                </td>
            </tr>
            """
        
        html += """
            </tbody>
        </table>
        </div>
        """
        return html


def model_call(state: AgentState,config: RunnableConfig | None = None) -> AgentState:
    # ✅ NEW (optional): access thread_id if you need it
    # thread_id = None
    # if config is not None:
    #     thread_id = config.configurable.get("thread_id")
    #     # e.g. for debugging:
    #     # print("Running model_call for thread:", thread_id)
    # elif isinstance(config, dict):
    #     thread_id = config.get("thread_id")  # Fallback for plain dict    
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
pretty_tools = PrettyToolNode(tools)
graph.add_node("tools", pretty_tools)
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue":"tools",
        "end":END
    }
)
graph.add_edge("tools","our_agent")
# ✅ NEW: compile with checkpointer
checkpointer = MemorySaver()
agent = graph.compile(checkpointer=checkpointer)

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
# while True:
#     user_query = input("🧭 Enter your travel query: ").strip()

#     if user_query.lower() == "exit":
#         break

#     conversation_history.append(HumanMessage(content=user_query))

#     result = agent.invoke({"messages": conversation_history})

#     final_message = result["messages"][-1]
#     final_message.pretty_print()

#     conversation_history.append(final_message)








