# VoyageCraft AI

Trip Planner Multi-Agent System with Guardrails and Human-in-the-Loop

A travel-planning application that uses a LangGraph multi-agent workflow to gather flight, hotel, weather, and budget information, refine the result through a supervisor and guardrail layer, and require human approval before finalizing the itinerary.

## Overview

This project combines:

- a FastAPI backend and web UI
- multi-agent travel reasoning with LangGraph
- input validation and safety guardrails
- MCP integrations for travel data and weather tools
- a human approval step before finalizing plans
- streaming progress updates for the user experience

The app is designed for a travel assistant that can accept natural-language trip requests and return a draft itinerary that can be reviewed and either approved or revised.

## Features

- Supervisor-based planning across specialist agents
- Structured workflow for:
  - flight recommendations
  - hotel recommendations
  - weather and forecast checks
  - budget feasibility analysis
  - itinerary drafting
- Guardrail layer to block unrelated or unsafe requests
- Human-in-the-loop approval flow with optional revision feedback
- PostgreSQL-backed LangGraph checkpointing
- FastAPI endpoints for both normal and streaming responses
- Frontend interface served from the project templates and static assets

## Tech Stack

- Python
- FastAPI
- LangGraph
- LangChain
- Ollama
- PostgreSQL
- MCP (Model Context Protocol)
- Tavily, AviationStack, and OpenWeather integrations

## Project Structure

```text
.
├── app.py                     # FastAPI entry point and HTTP routes
├── backend.py                 # LangGraph multi-agent workflow and orchestration
├── mcp_client.py              # MCP client for Tavily, AviationStack, and weather tools
├── custom_weather_mcp_server.py
│                             # OpenWeather MCP weather server
├── static/
│   ├── script.js
│   └── style.css
├── templates/
│   └── index.html
├── tests/
├── requirements.txt
├── Dockerfile
├── README.md
└── .env                      # local environment configuration (not committed)
```

## Prerequisites

Before running the project, make sure you have:

- Python 3.11+
- pip
- Ollama installed and running locally
- a local or remote PostgreSQL database for LangGraph checkpointing
- valid API keys for:
  - Tavily
  - AviationStack
  - OpenWeather
- the `uvx` command available for AviationStack MCP server setup

## Local Setup

1. Clone the repository

```bash
git clone <your-repo-url>
cd trip-planner-multi-agent-system-with-guardrails-hitl
```

2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root

```env
DATABASE_URL=postgresql://user:password@host:5432/dbname
TAVILY_API_KEY=your_tavily_key
AVIATION_STACK_API_KEY=your_aviationstack_key
OPENWEATHER_API_KEY=your_openweather_key
GROQ_API_KEY=your_groq_key
```

> Note: the current code uses Ollama by default for the LLM, so `GROQ_API_KEY` is optional unless you later switch to the Groq-based provider.

5. Start Ollama and make sure the model is available

```bash
ollama serve
ollama pull qwen2.5-coder:1.5b
```

6. Run the app

```bash
uvicorn app:app --host 0.0.0.0 --port 8020 --reload
```

Then open:

```text
http://localhost:8020
```

## API Endpoints

### Web UI

- `GET /` — renders the trip planner interface

### Travel planning

- `POST /api/travel` — submit a travel request and receive a generated result
- `GET /api/travel/stream` — stream progress events while the planner works
- `POST /api/travel/approve` — approve or reject the itinerary draft

### Health

- `GET /health` — returns the service health status

## Example Workflow

1. User enters a trip request such as:
   - "Plan a 5-day trip to Lisbon with a budget of $1200"
2. The supervisor decides which specialist agents are required.
3. Flight, hotel, weather, and budget agents produce structured suggestions.
4. The itinerary agent creates a draft plan.
5. The user reviews the itinerary and either approves it or provides feedback.
6. The workflow resumes and finalizes the plan.

## Human-in-the-Loop Approval

The approval flow is intentionally designed to pause before final output generation. The assistant presents a draft itinerary and asks for review. If the user approves, the final plan is created. If the user rejects or requests changes, the feedback is passed back into the workflow for revision.

## Troubleshooting

### Missing environment variables

Check that your `.env` file includes all required values and that you have restarted the app after editing it.

### `uvx` command not found

Install `uv` and ensure it is on your PATH:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then verify:

```bash
uvx --version
```

### Ollama connection issues

Confirm that Ollama is running locally and the required model exists:

```bash
ollama ps
ollama list
```

### PostgreSQL connection issues

The project expects a valid `DATABASE_URL` for `langgraph-checkpoint-postgres`. Ensure the database is reachable and that SSL is configured if needed.

## Notes

This project is suitable as a research or prototype travel assistant for multi-agent orchestration, human review, and guardrail-based planning. It is not a production booking engine and should be enhanced with additional validation, persistence, and business logic before being used in a live travel system.

## License

This project is licensed under the terms of the repository license.
