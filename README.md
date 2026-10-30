# Moodle

Autocorrect for how your message *sounds*, not just how it's spelled. Tonefix underlines phrases that may land badly
(blunt, passive-aggressive, vague, overly apologetic), explains why, and offers rewrites in the tone you pick.

## How it uses NVIDIA Nemotron on Nebius Token Factory

A cascade router sends each message to the cheapest model that can handle it:

| Step | Model | Job | When |
|---|---|---|---|
| 1 | Nemotron Nano | Fast tone check, returns compact JSON | Every check (after a pause in typing) |
| 2 | Nemotron Super | Writes 3 rewrites | Only if Nano's risk score is high |
| 3 | Nemotron 3 Ultra | Deep reasoning: sarcasm, implied meaning, thread context | Nano is unsure, or the user clicks "Deep check" |

Token Factory's OpenAI-compatible API keeps the calls fast and lets the app switch models per step with one config change.
The UI shows which models ran, how long each took, and the tokens used.

Token savers: 800 ms debounce, 3-word minimum, tiny `max_tokens` for Nano, and an in-memory result cache.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env      # add your API key and model IDs
uvicorn main:app --reload
```

Open http://localhost:8000. Check http://localhost:8000/api/health to confirm the config is picked up.

## Deploy on Nebius Serverless Endpoints

Build the included `Dockerfile`, push the image to a registry, and create a Serverless Endpoint on port 8000
with the variables from `.env.example` set as environment variables.

## Project layout

- `main.py` — FastAPI app and the Nano → Super → Ultra router
- `static/index.html` — single-page UI
