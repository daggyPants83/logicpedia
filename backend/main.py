from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
import uuid
from datetime import datetime
import os

app = FastAPI(title="Moral Reasoning Graph API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.environ.get("DB_PATH", "moral_graph.db")

# ─── DB INIT ──────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db()

# ── FRONTEND ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r") as f:
        return f.read()
:
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            statement TEXT NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('EMPIRICAL_FACT','LOGICAL_CONCLUSION','VALUE_ASSERTION','OBSERVED_PATTERN')),
            confidence TEXT NOT NULL DEFAULT 'SUPPORTED' CHECK(confidence IN ('ESTABLISHED','SUPPORTED','CONTESTED','SPECULATIVE')),
            created_by TEXT NOT NULL DEFAULT 'anonymous',
            created_at TEXT NOT NULL,
            is_deleted INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS edges (
            id TEXT PRIMARY KEY,
            node_a_id TEXT NOT NULL REFERENCES nodes(id),
            node_b_id TEXT NOT NULL REFERENCES nodes(id),
            relationship TEXT NOT NULL CHECK(relationship IN ('CAUSES','CONTRADICTS','PREREQUISITE_FOR','CORRELATES_WITH','LOGICALLY_IMPLIES','SUPPORTS','UNDERMINES')),
            direction TEXT NOT NULL DEFAULT 'DIRECTED' CHECK(direction IN ('DIRECTED','BIDIRECTIONAL')),
            strength REAL NOT NULL DEFAULT 0.5 CHECK(strength >= 0.0 AND strength <= 1.0),
            warrant TEXT NOT NULL,
            conditions TEXT,
            created_by TEXT NOT NULL DEFAULT 'anonymous',
            created_at TEXT NOT NULL,
            is_deleted INTEGER NOT NULL DEFAULT 0,
            CHECK(node_a_id != node_b_id)
        );

        CREATE TABLE IF NOT EXISTS challenges (
            id TEXT PRIMARY KEY,
            target_id TEXT NOT NULL,
            target_type TEXT NOT NULL CHECK(target_type IN ('NODE','EDGE')),
            ground TEXT NOT NULL CHECK(ground IN ('EMPIRICAL','LOGICAL','VALUE','SCOPE')),
            property_disputed TEXT NOT NULL,
            argument TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'OPEN' CHECK(status IN ('OPEN','RESOLVED_UPHELD','RESOLVED_REJECTED','SUPERSEDED')),
            resolution TEXT,
            created_by TEXT NOT NULL DEFAULT 'anonymous',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS evidence (
            id TEXT PRIMARY KEY,
            target_id TEXT NOT NULL,
            target_type TEXT NOT NULL CHECK(target_type IN ('NODE','EDGE')),
            source_type TEXT NOT NULL CHECK(source_type IN ('PEER_REVIEWED','OBSERVATIONAL','ANECDOTAL','LOGICAL_DERIVATION','EXPERT_CONSENSUS')),
            description TEXT NOT NULL,
            url TEXT,
            citation TEXT,
            created_by TEXT NOT NULL DEFAULT 'anonymous',
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

init_db()

# ─── MODELS ───────────────────────────────────────────────────────────────────

class NodeCreate(BaseModel):
    statement: str
    type: str
    confidence: str = "SUPPORTED"
    created_by: str = "anonymous"

class EdgeCreate(BaseModel):
    node_a_id: str
    node_b_id: str
    relationship: str
    direction: str = "DIRECTED"
    strength: float = 0.5
    warrant: str
    conditions: Optional[str] = None
    created_by: str = "anonymous"

class ChallengeCreate(BaseModel):
    target_id: str
    target_type: str
    ground: str
    property_disputed: str
    argument: str
    created_by: str = "anonymous"

class ChallengeResolve(BaseModel):
    status: str
    resolution: str

class EvidenceCreate(BaseModel):
    target_id: str
    target_type: str
    source_type: str
    description: str
    url: Optional[str] = None
    citation: Optional[str] = None
    created_by: str = "anonymous"

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def row_to_dict(row):
    return dict(row) if row else None

def now():
    return datetime.utcnow().isoformat()

def new_id():
    return str(uuid.uuid4())

# ─── NODES ────────────────────────────────────────────────────────────────────

@app.get("/nodes")
def list_nodes():
    conn = get_db()
    rows = conn.execute("SELECT * FROM nodes WHERE is_deleted=0 ORDER BY created_at DESC").fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]

@app.post("/nodes")
def create_node(data: NodeCreate):
    conn = get_db()
    node_id = new_id()
    conn.execute(
        "INSERT INTO nodes VALUES (?,?,?,?,?,?,0)",
        (node_id, data.statement, data.type, data.confidence, data.created_by, now())
    )
    conn.commit()
    row = conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
    conn.close()
    return row_to_dict(row)

@app.get("/nodes/{node_id}")
def get_node(node_id: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Node not found")
    node = row_to_dict(row)
    node["evidence"] = [row_to_dict(r) for r in conn.execute("SELECT * FROM evidence WHERE target_id=? AND target_type='NODE'", (node_id,)).fetchall()]
    node["challenges"] = [row_to_dict(r) for r in conn.execute("SELECT * FROM challenges WHERE target_id=? AND target_type='NODE'", (node_id,)).fetchall()]
    conn.close()
    return node

@app.delete("/nodes/{node_id}")
def delete_node(node_id: str):
    conn = get_db()
    conn.execute("UPDATE nodes SET is_deleted=1 WHERE id=?", (node_id,))
    conn.commit()
    conn.close()
    return {"deleted": True}

# ─── EDGES ────────────────────────────────────────────────────────────────────

@app.get("/edges")
def list_edges():
    conn = get_db()
    rows = conn.execute("SELECT * FROM edges WHERE is_deleted=0 ORDER BY created_at DESC").fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]

@app.post("/edges")
def create_edge(data: EdgeCreate):
    # Validate nodes exist
    conn = get_db()
    a = conn.execute("SELECT id FROM nodes WHERE id=? AND is_deleted=0", (data.node_a_id,)).fetchone()
    b = conn.execute("SELECT id FROM nodes WHERE id=? AND is_deleted=0", (data.node_b_id,)).fetchone()
    if not a or not b:
        raise HTTPException(404, "One or both nodes not found")
    edge_id = new_id()
    conn.execute(
        "INSERT INTO edges VALUES (?,?,?,?,?,?,?,?,?,?,0)",
        (edge_id, data.node_a_id, data.node_b_id, data.relationship,
         data.direction, data.strength, data.warrant, data.conditions,
         data.created_by, now())
    )
    conn.commit()
    row = conn.execute("SELECT * FROM edges WHERE id=?", (edge_id,)).fetchone()
    conn.close()
    return row_to_dict(row)

@app.get("/edges/{edge_id}")
def get_edge(edge_id: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM edges WHERE id=?", (edge_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Edge not found")
    edge = row_to_dict(row)
    edge["evidence"] = [row_to_dict(r) for r in conn.execute("SELECT * FROM evidence WHERE target_id=? AND target_type='EDGE'", (edge_id,)).fetchall()]
    edge["challenges"] = [row_to_dict(r) for r in conn.execute("SELECT * FROM challenges WHERE target_id=? AND target_type='EDGE'", (edge_id,)).fetchall()]
    conn.close()
    return edge

@app.delete("/edges/{edge_id}")
def delete_edge(edge_id: str):
    conn = get_db()
    conn.execute("UPDATE edges SET is_deleted=1 WHERE id=?", (edge_id,))
    conn.commit()
    conn.close()
    return {"deleted": True}

# ─── CHALLENGES ───────────────────────────────────────────────────────────────

@app.post("/challenges")
def create_challenge(data: ChallengeCreate):
    conn = get_db()
    cid = new_id()
    conn.execute(
        "INSERT INTO challenges VALUES (?,?,?,?,?,?,?,?,?,?)",
        (cid, data.target_id, data.target_type, data.ground,
         data.property_disputed, data.argument, "OPEN", None,
         data.created_by, now())
    )
    conn.commit()
    row = conn.execute("SELECT * FROM challenges WHERE id=?", (cid,)).fetchone()
    conn.close()
    return row_to_dict(row)

@app.patch("/challenges/{challenge_id}/resolve")
def resolve_challenge(challenge_id: str, data: ChallengeResolve):
    conn = get_db()
    conn.execute(
        "UPDATE challenges SET status=?, resolution=? WHERE id=?",
        (data.status, data.resolution, challenge_id)
    )
    conn.commit()
    row = conn.execute("SELECT * FROM challenges WHERE id=?", (challenge_id,)).fetchone()
    conn.close()
    return row_to_dict(row)

# ─── EVIDENCE ─────────────────────────────────────────────────────────────────

@app.post("/evidence")
def create_evidence(data: EvidenceCreate):
    conn = get_db()
    eid = new_id()
    conn.execute(
        "INSERT INTO evidence VALUES (?,?,?,?,?,?,?,?,?)",
        (eid, data.target_id, data.target_type, data.source_type,
         data.description, data.url, data.citation, data.created_by, now())
    )
    conn.commit()
    row = conn.execute("SELECT * FROM evidence WHERE id=?", (eid,)).fetchone()
    conn.close()
    return row_to_dict(row)

# ─── GRAPH (full export for visualisation) ────────────────────────────────────

@app.get("/graph")
def get_graph():
    conn = get_db()
    nodes = [row_to_dict(r) for r in conn.execute("SELECT * FROM nodes WHERE is_deleted=0").fetchall()]
    edges = [row_to_dict(r) for r in conn.execute("SELECT * FROM edges WHERE is_deleted=0").fetchall()]
    # attach challenge counts
    for n in nodes:
        count = conn.execute("SELECT COUNT(*) FROM challenges WHERE target_id=? AND status='OPEN'", (n["id"],)).fetchone()[0]
        n["open_challenges"] = count
    for e in edges:
        count = conn.execute("SELECT COUNT(*) FROM challenges WHERE target_id=? AND status='OPEN'", (e["id"],)).fetchone()[0]
        e["open_challenges"] = count
    conn.close()
    return {"nodes": nodes, "edges": edges}

# ─── SEED DATA ────────────────────────────────────────────────────────────────

@app.post("/seed")
def seed():
    """Seed the database with the pig suffering example chain from our conversation."""
    conn = get_db()
    existing = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    if existing > 0:
        conn.close()
        return {"message": "Already seeded"}

    t = now()
    nodes = [
        (new_id(), "Pigs have a complex nervous system capable of processing pain signals", "EMPIRICAL_FACT", "ESTABLISHED", "seed", t),
        (new_id(), "Pigs demonstrate distress behaviours under harmful stimuli comparable to mammals with known pain experience", "OBSERVED_PATTERN", "SUPPORTED", "seed", t),
        (new_id(), "Pigs experience suffering", "LOGICAL_CONCLUSION", "SUPPORTED", "seed", t),
        (new_id(), "Industrial pig farming subjects pigs to conditions that cause sustained distress", "OBSERVED_PATTERN", "SUPPORTED", "seed", t),
        (new_id(), "Industrial pig farming causes large scale animal suffering", "LOGICAL_CONCLUSION", "SUPPORTED", "seed", t),
        (new_id(), "Causing unnecessary suffering to sentient creatures is morally wrong", "VALUE_ASSERTION", "CONTESTED", "seed", t),
        (new_id(), "Industrial pig farming as currently practised is morally unjustifiable unless suffering is eliminated", "LOGICAL_CONCLUSION", "CONTESTED", "seed", t),
    ]
    conn.executemany("INSERT INTO nodes VALUES (?,?,?,?,?,?,0)", nodes)

    ids = [n[0] for n in nodes]
    edges = [
        (new_id(), ids[0], ids[2], "SUPPORTS", "DIRECTED", 0.8,
         "Neurological capacity for pain is the biological basis for pain experience", None, "seed", t),
        (new_id(), ids[1], ids[2], "LOGICALLY_IMPLIES", "DIRECTED", 0.75,
         "Behavioural evidence of distress in animals with pain-capable nervous systems implies subjective experience", None, "seed", t),
        (new_id(), ids[2], ids[4], "PREREQUISITE_FOR", "DIRECTED", 0.9,
         "Suffering requires a subject capable of suffering; established capacity is required before scale claim", None, "seed", t),
        (new_id(), ids[3], ids[4], "CAUSES", "DIRECTED", 0.85,
         "Documented factory farming conditions directly produce the distress behaviours established as suffering indicators", None, "seed", t),
        (new_id(), ids[4], ids[6], "LOGICALLY_IMPLIES", "DIRECTED", 0.8,
         "If suffering occurs at scale and suffering is wrong, the practice producing it is unjustifiable", None, "seed", t),
        (new_id(), ids[5], ids[6], "PREREQUISITE_FOR", "DIRECTED", 1.0,
         "The value assertion that suffering is wrong is required for the moral conclusion to follow", None, "seed", t),
    ]
    conn.executemany("INSERT INTO edges VALUES (?,?,?,?,?,?,?,?,?,?,0)", edges)
    conn.commit()
    conn.close()
    return {"message": "Seeded", "nodes": len(nodes), "edges": len(edges)}
