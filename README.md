# Moral Reasoning Graph

A collaborative tool for making chains of moral reasoning explicit, visible, and challengeable.

Built from a conversation about epistemics, cognitive dissonance, the USSR, pig farming, and whether LLMs are just sheep.


---

## What it does

- Create **nodes** — empirical facts, logical conclusions, value assertions, observed patterns
- Connect them with **edges** that require an explicit **warrant** (why does this connection hold?)
- **Challenge** any node or edge, declaring whether your dispute is empirical, logical, value-based, or about scope
- Add **evidence** to support nodes and connections
- Visualise the whole graph — drag nodes, click to explore

The key insight: you cannot connect two ideas without stating *why* they connect. That single constraint makes most bad reasoning visible.

---

## Run locally

```bash
# Install Python dependencies
cd backend
pip install -r requirements.txt

# Start backend
uvicorn main:app --reload --port 8000

# Open frontend in browser
open frontend/index.html
# or serve it:
cd frontend && python -m http.server 3000
```

Then open http://localhost:3000 and click **Load Example** to seed the pig suffering chain.

---

## Deploy with Docker (recommended)

Requirements: a Linux server with Docker and Docker Compose installed.

```bash
# Clone or copy this folder to your server
scp -r moral-graph user@yourserver:~/

# On your server:
cd moral-graph
docker-compose up -d
```

The app will be at http://yourserver (port 80).

### Cheap server options
- **Railway** — push to GitHub, connect repo, deploy. Free tier available.
- **Render** — similar, free tier available, good for this scale.
- **DigitalOcean Droplet** — $6/month, full control, use Docker Compose above.
- **Fly.io** — free tier, good CLI, handles Docker natively.

---

## Deployment on Railway (easiest)

1. Push this folder to a GitHub repo
2. Go to railway.app, create new project from GitHub repo
3. Railway auto-detects Docker Compose
4. Add a volume for `/data` in the Railway dashboard
5. Done — you get a public URL

---

## Architecture

```
frontend/index.html    — single file web app (D3.js graph visualisation)
backend/main.py        — FastAPI + SQLite REST API
backend/requirements.txt
docker-compose.yml     — nginx (frontend) + uvicorn (backend)
nginx.conf             — proxies /api/* to backend
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /graph | Full graph for visualisation |
| GET/POST | /nodes | List or create nodes |
| GET | /nodes/:id | Node with evidence and challenges |
| GET/POST | /edges | List or create edges |
| GET | /edges/:id | Edge with evidence and challenges |
| POST | /challenges | Raise a challenge |
| PATCH | /challenges/:id/resolve | Resolve a challenge |
| POST | /evidence | Add evidence |
| POST | /seed | Load example data |

---

## The object model

**Node** — a claim. Has a type (empirical/logical/value/pattern) and a confidence level.

**Edge** — a connection between two nodes. Requires a *warrant*: the explicit statement of why this connection holds. This is the Toulmin insight made structural.

**Challenge** — a dispute against a node or edge. Must declare its *ground*: empirical, logical, value, or scope. This makes the nature of disagreement legible.

**Evidence** — supporting material for a node or edge, typed by source quality.

Nothing is deleted. Everything is flagged. The history of arguments is itself data.
