# Travel AI Assistant

A LangGraph-powered conversational travel coordinator designed to plan trips by searching flights and hotels. The assistant leverages LLMs (via OpenRouter or Groq), the Amadeus Self-Service API for flight search, and Google Hotels via SerpApi for accommodation search.

It is available in two modes:
1. **Standalone CLI Mode** (`new_react.py`): An interactive, terminal-based chat interface.
2. **FastAPI Web Server Mode** (`backend/app/main.py`): A WebSocket-enabled backend server that streams agent decisions, tool execution, and results in real time.

---

## Project Structure

```text
travel_ai_assistant/
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── graph.py       # LangGraph state machine, coordinator agent, and tools
│       ├── main.py        # FastAPI server and WebSocket endpoint handler
│       └── session.py     # Task session manager for request interruption
├── new_react.py           # Standalone terminal-based CLI assistant
├── requirements.txt       # Core project package requirements
├── test.html              # Simple HTML page to test WebSocket connections
└── .env                   # Local environment variables (API keys)
```

---

## Prerequisites & API Keys

To run the application, you need to configure a `.env` file in the root directory. Copy or create a `.env` file with the following variables:

```env
# 1. LLM API Keys
OPENROUTER_API_KEY=your_openrouter_api_key
XAI_API_KEY=your_groq_or_xai_api_key

# 2. Amadeus Flight Search API Credentials (Test Environment)
AMADEUS_API_KEY=your_amadeus_client_id
AMADEUS_API_SECRET=your_amadeus_client_secret

# 3. SerpApi Google Hotels API Key
SERPAPI_API_KEY=your_serpapi_api_key
```

### Key Descriptions & Acquisition:
* **OpenRouter API Key (`OPENROUTER_API_KEY`)**: Used by the backend server (`graph.py`) to query LLMs using `openrouter/auto:free`. Register at [OpenRouter.ai](https://openrouter.ai/).
* **Groq API Key (`XAI_API_KEY` / `GROQ_API_KEY`)**: Used by the CLI mode (`new_react.py`) to query Groq. Register at [console.groq.com](https://console.groq.com/).
* **Amadeus Credentials (`AMADEUS_API_KEY` & `AMADEUS_API_SECRET`)**: Used to fetch flight prices and schedules. Register for a free account at [Amadeus for Developers](https://developers.amadeus.com/) and create a project to obtain **Test** environment keys.
* **SerpApi Key (`SERPAPI_API_KEY`)**: Used to scrape Google Hotels search results for hotel availability, prices, and ratings. Register at [SerpApi.com](https://serpapi.com/).

---

## Setup Instructions

### 1. Create a Virtual Environment
From the root directory of the project, run:
```bash
python3 -m venv venv
```

### 2. Activate the Virtual Environment
* **macOS / Linux**:
  ```bash
  source venv/bin/activate
  ```
* **Windows (Command Prompt)**:
  ```cmd
  venv\Scripts\activate
  ```
* **Windows (PowerShell)**:
  ```powershell
  venv\Scripts\Activate.ps1
  ```

### 3. Install Dependencies
Run the following command to install the required libraries:
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install langgraph langchain-openai langchain-groq fastapi uvicorn requests google-search-results python-dotenv IPython
```

---

## How to Run

### Option A: Standalone CLI Assistant
To run the interactive console assistant:
```bash
python new_react.py
```
Type your query (e.g., *“Find flights from LHR to CDG on 2026-10-15 and hotels in Paris”*) and press **Enter**. Type `exit` to quit.

### Option B: FastAPI Backend Server
To run the web application backend with real-time WebSocket communication:
1. Start the development server using Uvicorn:
   ```bash
   uvicorn backend.app.main:app --reload
   ```
2. The backend will start on `http://127.0.0.1:8000`.
3. To connect and test the assistant, open the `test.html` file or your styled frontend `test.html` in a web browser. Ensure the browser's developer console (F12) is open to view stream logs.
