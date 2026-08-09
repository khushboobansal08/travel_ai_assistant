# import session
# import graph
# from session import session_manager
# from graph import agent
from . import session
from . import graph
from .session import session_manager
from .graph import agent
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
from langchain_core.messages import HumanMessage

app = FastAPI()



async def run_langgraph_stream(websocket: WebSocket, thread_id: str, user_content: str):
    """Runs LangGraph and streams ALL output to WebSocket."""
    config = {"configurable": {"thread_id": thread_id}}
    async for event in agent.astream(
        {"messages": [HumanMessage(content=user_content)]},
        config=config,
        stream_mode="values"
    ):
        msg = event["messages"][-1]
        # ✅ FILTER: Skip user echo (same as input)
        if msg.content == user_content or msg.content.strip() == "":
            continue
            
        # ✅ FILTER: Skip empty stream markers
        if msg.content.startswith('{"type":"stream"') or not msg.content.strip():
            continue
            
        # ✅ ONLY send FINAL clean content
        await websocket.send_json({
            "role": "ai",
            "content": msg.content.strip()
        })    
        # await websocket.send_json({
        #     "type": "stream",
        #     "content": msg.content,
        #     "role": getattr(msg, "type", "unknown")
         

@app.websocket("/ws/{thread_id}")
async def websocket_endpoint(websocket: WebSocket, thread_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            # HANDLE INTERRUPT FIRST
            if msg_type == "interrupt":
                session_manager.cancel(thread_id)
                await websocket.send_json({"role": "ai", 
                    "content": "🛑 Cancelled. What next?"})
                continue
            
            # NORMAL MESSAGE (only if not interrupt)
            content = data["content"]  # user message like "find flights"
            
            # CREATE ONE TASK
            task = asyncio.create_task(
                run_langgraph_stream(websocket, thread_id, content)
            )
            
            # REGISTER + WAIT
            session_manager.register_task(thread_id, task)
            await task  # waits for stream to finish
            
    except WebSocketDisconnect:
        session_manager.cleanup(thread_id)
