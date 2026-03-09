from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
import uuid
from datetime import datetime
import os
import json

app = FastAPI(
    title="Logicpedia — Structured Reasoning Graph",
    description="""
## The Universal Reasoning Graph

A single, infinite graph of structured knowledge. Every claim is a typed triple.
Every connection requires a warrant. Nothing is deleted — only superseded.

### For LLMs
This API is designed to be navigated and extended by language models.
- Use `/vocab` to discover available terms before creating nodes
- Use `/llm/context` to get a full reasoning context dump
- Use `/llm/propose` to add nodes, edges and evidence in one call
- Use `/path` to find reasoning chains between concepts
- All endpoints return machine-readable structured data

### Claim Structure
Every node is a triple: `[subject] [predicate] [object]`
- Subjects and objects are drawn from the **entity vocabulary**
- Predicates are drawn from the **predicate vocabulary**
- New vocabulary terms can be proposed via `/vocab/propose`

### Edge Warrants
Every edge must cite a **warrant node** or **warrant text** that
justifies the logical connection. This forces the reasoning
structure to be explicit and self-referential.
""",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.environ.get("DB_PATH", "logicpedia.db")

# ── DB ──────────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def now():
    return datetime.utcnow().isoformat()

def new_id():
    return str(uuid.uuid4())

def row_to_dict(row):
    return dict(row) if row else None

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS vocab_entities (
            id TEXT PRIMARY KEY,
            term TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            description TEXT,
            proposed_by TEXT NOT NULL DEFAULT 'system',
            status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE','DEPRECATED','PROPOSED')),
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS vocab_predicates (
            id TEXT PRIMARY KEY,
            term TEXT NOT NULL UNIQUE,
            english TEXT NOT NULL,
            direction_hint TEXT NOT NULL DEFAULT 'FORWARD',
            domain TEXT,
            range_hint TEXT,
            description TEXT,
            status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE','DEPRECATED','PROPOSED')),
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL REFERENCES vocab_entities(id),
            predicate_id TEXT NOT NULL REFERENCES vocab_predicates(id),
            object_id TEXT NOT NULL REFERENCES vocab_entities(id),
            statement TEXT NOT NULL,
            node_type TEXT NOT NULL CHECK(node_type IN ('EMPIRICAL','LOGICAL','VALUE','DEFINITIONAL','OBSERVED')),
            confidence TEXT NOT NULL DEFAULT 'SUPPORTED' CHECK(confidence IN ('ESTABLISHED','SUPPORTED','CONTESTED','SPECULATIVE','REFUTED')),
            domain_tags TEXT NOT NULL DEFAULT '[]',
            created_by TEXT NOT NULL DEFAULT 'anonymous',
            created_at TEXT NOT NULL,
            is_deleted INTEGER NOT NULL DEFAULT 0,
            superseded_by TEXT
        );

        CREATE TABLE IF NOT EXISTS edges (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES nodes(id),
            target_id TEXT NOT NULL REFERENCES nodes(id),
            relationship TEXT NOT NULL CHECK(relationship IN (
                'SUPPORTS','UNDERMINES','REQUIRES','CONTRADICTS',
                'CAUSES','IMPLIES','CORRELATES_WITH','DEFINES','EXEMPLIFIES','PREVENTS'
            )),
            warrant_node_id TEXT,
            warrant_text TEXT,
            strength REAL NOT NULL DEFAULT 0.5 CHECK(strength >= 0 AND strength <= 1),
            conditions TEXT,
            created_by TEXT NOT NULL DEFAULT 'anonymous',
            created_at TEXT NOT NULL,
            is_deleted INTEGER NOT NULL DEFAULT 0,
            CHECK(source_id != target_id)
        );

        CREATE TABLE IF NOT EXISTS challenges (
            id TEXT PRIMARY KEY,
            target_id TEXT NOT NULL,
            target_type TEXT NOT NULL CHECK(target_type IN ('NODE','EDGE')),
            ground TEXT NOT NULL CHECK(ground IN ('EMPIRICAL','LOGICAL','VALUE','SCOPE','DEFINITION')),
            property_disputed TEXT NOT NULL,
            argument TEXT NOT NULL,
            counter_node_id TEXT,
            status TEXT NOT NULL DEFAULT 'OPEN' CHECK(status IN ('OPEN','RESOLVED_UPHELD','RESOLVED_REJECTED','SUPERSEDED')),
            resolution TEXT,
            created_by TEXT NOT NULL DEFAULT 'anonymous',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS evidence (
            id TEXT PRIMARY KEY,
            target_id TEXT NOT NULL,
            target_type TEXT NOT NULL CHECK(target_type IN ('NODE','EDGE')),
            source_type TEXT NOT NULL CHECK(source_type IN (
                'PEER_REVIEWED','META_ANALYSIS','SYSTEMATIC_REVIEW',
                'OBSERVATIONAL','EXPERT_CONSENSUS','LOGICAL_DERIVATION','ANECDOTAL'
            )),
            description TEXT NOT NULL,
            url TEXT,
            citation TEXT,
            year INTEGER,
            created_by TEXT NOT NULL DEFAULT 'anonymous',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS views (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            focus_node_id TEXT,
            pinned_node_ids TEXT NOT NULL DEFAULT '[]',
            depth INTEGER NOT NULL DEFAULT 2,
            created_by TEXT NOT NULL DEFAULT 'anonymous',
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()
    _seed_vocab()
    _seed_graph()

# ── VOCAB SEED ───────────────────────────────────────────────────────────────

def _seed_vocab():
    conn = get_db()
    existing = conn.execute("SELECT COUNT(*) FROM vocab_entities").fetchone()[0]
    if existing > 0:
        conn.close()
        return

    t = now()

    entities = [
        # AGENTS
        ("pig", "AGENT", "Domestic pig (Sus scrofa domesticus)"),
        ("human", "AGENT", "Human being (Homo sapiens)"),
        ("animal", "AGENT", "Non-human animal"),
        ("sentient_creature", "AGENT", "Any being capable of subjective experience"),
        ("corporation", "AGENT", "A legal corporate entity"),
        ("government", "AGENT", "State or governmental body"),
        ("consumer", "AGENT", "Individual making purchasing decisions"),
        ("worker", "AGENT", "Individual performing labour"),
        ("child", "AGENT", "Human being below age of majority"),
        ("future_generation", "AGENT", "People not yet born"),
        # CONCEPTS
        ("suffering", "CONCEPT", "Subjective experience of pain or distress"),
        ("pain", "CONCEPT", "Nociceptive signal processing resulting in aversive experience"),
        ("consciousness", "CONCEPT", "Subjective, first-person experience"),
        ("moral_worth", "CONCEPT", "The property of deserving moral consideration"),
        ("moral_responsibility", "CONCEPT", "Obligation to act or refrain based on ethical principles"),
        ("rights", "CONCEPT", "Entitlements that constrain how an entity may be treated"),
        ("welfare", "CONCEPT", "Overall quality of life and freedom from suffering"),
        ("autonomy", "CONCEPT", "Capacity for self-determination"),
        ("equality", "CONCEPT", "Equal moral consideration of equal interests"),
        ("justice", "CONCEPT", "Fair distribution of benefits and burdens"),
        ("harm", "CONCEPT", "Damage to wellbeing or interests"),
        ("benefit", "CONCEPT", "Positive contribution to wellbeing or interests"),
        ("necessity", "CONCEPT", "The property of being required for a goal"),
        ("consent", "CONCEPT", "Voluntary agreement to an action"),
        ("freedom", "CONCEPT", "Absence of coercion or constraint"),
        ("dignity", "CONCEPT", "Inherent worth deserving respect"),
        ("truth", "CONCEPT", "Correspondence between belief and reality"),
        ("knowledge", "CONCEPT", "Justified true belief"),
        ("power", "CONCEPT", "Capacity to effect outcomes in the world"),
        # PROCESSES
        ("industrial_farming", "PROCESS", "Large-scale, intensive animal agriculture"),
        ("factory_farming", "PROCESS", "Confined, high-density industrial animal production"),
        ("selective_breeding", "PROCESS", "Artificial selection for heritable traits"),
        ("slaughter", "PROCESS", "Killing of animals for food"),
        ("eating_meat", "PROCESS", "Consumption of animal flesh"),
        ("eating_bacon", "PROCESS", "Consumption of cured pork products"),
        ("capitalism", "PROCESS", "Economic system based on private ownership and market exchange"),
        ("democracy", "PROCESS", "Political system based on popular representation"),
        ("regulation", "PROCESS", "Government imposition of rules on actors"),
        ("evolution", "PROCESS", "Change in heritable traits over generations"),
        ("reproduction", "PROCESS", "Biological generation of offspring"),
        ("communication", "PROCESS", "Exchange of information between agents"),
        ("education", "PROCESS", "Transmission of knowledge and skills"),
        ("taxation", "PROCESS", "Compulsory government levy on income or wealth"),
        ("war", "PROCESS", "Armed conflict between organised groups"),
        ("cooperation", "PROCESS", "Joint action toward shared goals"),
        # STATES
        ("confinement", "STATE", "Restriction of movement below natural range"),
        ("distress", "STATE", "Aversive psychological or physical state"),
        ("sentience", "STATE", "Capacity for subjective experience"),
        ("sapience", "STATE", "Capacity for higher reasoning and self-awareness"),
        ("extinction", "STATE", "Complete elimination of a species"),
        ("poverty", "STATE", "Insufficient resources for basic needs"),
        ("inequality", "STATE", "Unequal distribution of resources or power"),
        ("abundance", "STATE", "Resources substantially exceeding basic needs"),
        ("stability", "STATE", "Resistance to disruptive change"),
        # PROPERTIES
        ("nervous_system", "PROPERTY", "Neural architecture capable of signal processing"),
        ("nociceptors", "PROPERTY", "Pain-sensing nerve endings"),
        ("cognitive_complexity", "PROPERTY", "Capacity for complex thought and problem-solving"),
        ("social_behaviour", "PROPERTY", "Capacity for complex social interaction"),
        ("nutritional_value", "PROPERTY", "Contribution to human dietary needs"),
        ("environmental_impact", "PROPERTY", "Effect on ecosystems and climate"),
        ("greenhouse_gas_emissions", "PROPERTY", "Contribution to atmospheric warming gases"),
        ("land_use", "PROPERTY", "Area of land required for a process"),
        ("water_use", "PROPERTY", "Volume of fresh water consumed"),
        # SYSTEMS
        ("food_system", "SYSTEM", "The totality of food production and consumption"),
        ("ecosystem", "SYSTEM", "Interacting community of organisms and environment"),
        ("climate", "SYSTEM", "Global atmospheric and weather system"),
        ("economy", "SYSTEM", "System of production, distribution and consumption"),
        ("legal_system", "SYSTEM", "Body of rules enforced by state authority"),
        ("moral_framework", "SYSTEM", "Systematic approach to ethical evaluation"),
        ("utilitarian_ethics", "SYSTEM", "Ethical framework maximising aggregate welfare"),
        ("deontological_ethics", "SYSTEM", "Ethics based on duties and rights regardless of outcome"),
        ("virtue_ethics", "SYSTEM", "Ethics based on character and human flourishing"),
        ("democracy_system", "SYSTEM", "Political system of popular representation and rule"),
        ("market", "SYSTEM", "Decentralised system of voluntary exchange"),
    ]

    for term, cat, desc in entities:
        conn.execute(
            "INSERT INTO vocab_entities VALUES (?,?,?,?,?,?,?)",
            (new_id(), term, cat, desc, 'system', 'ACTIVE', t)
        )

    predicates = [
        ("HAS_PROPERTY", "has the property of", "FORWARD", None, "PROPERTY", "Subject possesses the stated property"),
        ("IS_CAPABLE_OF", "is capable of", "FORWARD", None, None, "Subject has the capacity for the stated process or state"),
        ("CAUSES", "causes", "FORWARD", None, None, "Subject brings about the object"),
        ("REQUIRES", "requires", "FORWARD", None, None, "Subject cannot exist without the object"),
        ("CONTRADICTS", "contradicts", "SYMMETRIC", None, None, "Subject and object cannot both be true"),
        ("IS_INSTANCE_OF", "is an instance of", "FORWARD", None, None, "Subject is a specific case of the object category"),
        ("IS_MORALLY_EQUIVALENT_TO", "is morally equivalent to", "SYMMETRIC", None, None, "Subject and object have the same moral status"),
        ("INCREASES", "increases", "FORWARD", None, None, "Subject causes the object to increase"),
        ("DECREASES", "decreases", "FORWARD", None, None, "Subject causes the object to decrease"),
        ("DEPENDS_ON", "depends on", "FORWARD", None, None, "Subject's existence or function depends on object"),
        ("PRODUCES", "produces", "FORWARD", None, None, "Subject generates or outputs the object"),
        ("PREVENTS", "prevents", "FORWARD", None, None, "Subject stops the object from occurring"),
        ("IMPLIES", "implies", "FORWARD", None, None, "If subject is true, object must be true"),
        ("CORRELATES_WITH", "correlates with", "SYMMETRIC", None, None, "Subject and object co-occur without established causation"),
        ("IS_JUSTIFIED_BY", "is justified by", "FORWARD", None, None, "Subject's permissibility is grounded in object"),
        ("VIOLATES", "violates", "FORWARD", None, None, "Subject transgresses or is incompatible with object"),
        ("CONSTITUTES", "constitutes", "FORWARD", None, None, "Subject makes up or defines object"),
        ("UNDERMINES", "undermines", "FORWARD", None, None, "Subject weakens or destabilises object"),
        ("SUPPORTS", "supports", "FORWARD", None, None, "Subject provides evidence or justification for object"),
        ("IS_NECESSARY_FOR", "is necessary for", "FORWARD", None, None, "Object cannot occur without subject"),
        ("IS_SUFFICIENT_FOR", "is sufficient for", "FORWARD", None, None, "Subject alone guarantees object"),
        ("EXPERIENCES", "experiences", "FORWARD", "AGENT", "CONCEPT", "Subject has subjective experience of object"),
        ("HAS_INTEREST_IN", "has an interest in", "FORWARD", "AGENT", None, "Subject has a stake in or is affected by object"),
        ("IS_HARMED_BY", "is harmed by", "FORWARD", "AGENT", None, "Subject's welfare is diminished by object"),
        ("IS_DEFINED_BY", "is defined by", "FORWARD", None, None, "Subject's nature is constituted by object"),
        ("CONTRIBUTES_TO", "contributes to", "FORWARD", None, None, "Subject partially causes or enables object"),
        ("IS_PRECONDITION_FOR", "is a precondition for", "FORWARD", None, None, "Subject must be true for object to be possible"),
        ("ENABLES", "enables", "FORWARD", None, None, "Subject makes object possible"),
        ("LIMITS", "limits", "FORWARD", None, None, "Subject constrains or restricts object"),
        ("TRADES_OFF_AGAINST", "trades off against", "SYMMETRIC", None, None, "Increasing subject reduces object and vice versa"),
    ]

    for term, eng, dirn, dom, rng, desc in predicates:
        conn.execute(
            "INSERT INTO vocab_predicates VALUES (?,?,?,?,?,?,?,?,?)",
            (new_id(), term, eng, dirn, dom, rng, desc, 'ACTIVE', t)
        )

    conn.commit()
    conn.close()

# ── GRAPH SEED ────────────────────────────────────────────────────────────────

def _seed_graph():
    conn = get_db()
    existing = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    if existing > 0:
        conn.close()
        return

    def e(term):
        r = conn.execute("SELECT id FROM vocab_entities WHERE term=?", (term,)).fetchone()
        if not r:
            raise ValueError(f"Entity not found: {term}")
        return r["id"]

    def p(term):
        r = conn.execute("SELECT id, english FROM vocab_predicates WHERE term=?", (term,)).fetchone()
        if not r:
            raise ValueError(f"Predicate not found: {term}")
        return r["id"], r["english"]

    def mk_node(subj, pred, obj, ntype, conf, tags, by="seed"):
        pid, peng = p(pred)
        statement = f"{subj} {peng} {obj}".replace("_", " ")
        nid = new_id()
        conn.execute(
            "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (nid, e(subj), pid, e(obj), statement, ntype, conf, json.dumps(tags), by, now(), 0, None)
        )
        return nid

    def mk_edge(src, tgt, rel, wt=None, strength=0.7, by="seed"):
        eid = new_id()
        conn.execute(
            "INSERT INTO edges VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (eid, src, tgt, rel, None, wt, strength, None, by, now(), 0)
        )
        return eid

    # ── Domain: Animal biology & consciousness ────────────────────────────
    n_pig_ns     = mk_node("pig","HAS_PROPERTY","nervous_system","EMPIRICAL","ESTABLISHED",["biology","neuroscience"])
    n_pig_noci   = mk_node("pig","HAS_PROPERTY","nociceptors","EMPIRICAL","ESTABLISHED",["biology","neuroscience"])
    n_pig_cog    = mk_node("pig","HAS_PROPERTY","cognitive_complexity","EMPIRICAL","ESTABLISHED",["biology","cognition"])
    n_pig_social = mk_node("pig","HAS_PROPERTY","social_behaviour","EMPIRICAL","ESTABLISHED",["biology","cognition"])
    n_pig_sent   = mk_node("pig","IS_CAPABLE_OF","sentience","LOGICAL","SUPPORTED",["biology","cognition","ethics"])
    n_pig_suffer = mk_node("pig","IS_CAPABLE_OF","suffering","LOGICAL","SUPPORTED",["biology","ethics"])
    n_pig_dist   = mk_node("pig","EXPERIENCES","distress","OBSERVED","ESTABLISHED",["biology","animal_welfare"])
    n_pig_pain   = mk_node("pig","EXPERIENCES","pain","OBSERVED","ESTABLISHED",["biology","animal_welfare"])

    mk_edge(n_pig_ns, n_pig_sent, "SUPPORTS", "A complex nervous system is the physical substrate of sentience", 0.85)
    mk_edge(n_pig_noci, n_pig_suffer, "SUPPORTS", "Nociceptors are the necessary sensory basis for pain experience", 0.9)
    mk_edge(n_pig_cog, n_pig_sent, "SUPPORTS", "Cognitive complexity is positively correlated with depth of subjective experience", 0.75)
    mk_edge(n_pig_sent, n_pig_suffer, "IMPLIES", "Sentience — the capacity for experience — entails the capacity for suffering", 0.9)
    mk_edge(n_pig_suffer, n_pig_dist, "CAUSES", "Suffering capacity under aversive conditions produces observed distress states", 0.9)
    mk_edge(n_pig_pain, n_pig_dist, "CAUSES", "Experienced pain is a sufficient cause of distress", 0.95)

    # ── Domain: Sentience & moral worth ──────────────────────────────────
    n_sent_mw    = mk_node("sentient_creature","HAS_PROPERTY","moral_worth","VALUE","CONTESTED",["ethics","moral_philosophy"])
    n_suffer_mw  = mk_node("suffering","IS_SUFFICIENT_FOR","moral_worth","VALUE","CONTESTED",["ethics","moral_philosophy"])
    n_sent_int   = mk_node("sentient_creature","HAS_INTEREST_IN","welfare","VALUE","SUPPORTED",["ethics","moral_philosophy"])
    n_harm_moral = mk_node("harm","VIOLATES","moral_framework","VALUE","CONTESTED",["ethics","moral_philosophy"])
    n_consent_j  = mk_node("consent","IS_NECESSARY_FOR","justice","VALUE","SUPPORTED",["ethics","political_philosophy"])
    n_animal_r   = mk_node("animal","HAS_INTEREST_IN","rights","VALUE","CONTESTED",["ethics","animal_rights"])

    mk_edge(n_pig_sent, n_sent_mw, "SUPPORTS", "Sentience is the widely-cited threshold for moral considerability", 0.8)
    mk_edge(n_pig_suffer, n_sent_int, "IMPLIES", "Capacity for suffering grounds welfare interests by definition", 0.85)
    mk_edge(n_sent_int, n_sent_mw, "IMPLIES", "Having welfare interests is both necessary and sufficient for moral considerability", 0.8)
    mk_edge(n_sent_mw, n_animal_r, "IMPLIES", "Moral worth grounds rights claims; without moral worth, rights have no basis", 0.75)

    # ── Domain: Factory farming conditions ───────────────────────────────
    n_ff_conf    = mk_node("factory_farming","CAUSES","confinement","EMPIRICAL","ESTABLISHED",["agriculture","animal_welfare"])
    n_ff_dist    = mk_node("factory_farming","CAUSES","distress","EMPIRICAL","ESTABLISHED",["agriculture","animal_welfare"])
    n_ff_harm    = mk_node("factory_farming","CAUSES","harm","LOGICAL","SUPPORTED",["agriculture","ethics"])
    n_if_env     = mk_node("industrial_farming","HAS_PROPERTY","environmental_impact","EMPIRICAL","ESTABLISHED",["agriculture","climate"])
    n_if_ghg     = mk_node("industrial_farming","HAS_PROPERTY","greenhouse_gas_emissions","EMPIRICAL","ESTABLISHED",["agriculture","climate"])
    n_if_land    = mk_node("industrial_farming","HAS_PROPERTY","land_use","EMPIRICAL","ESTABLISHED",["agriculture","environment"])
    n_if_water   = mk_node("industrial_farming","HAS_PROPERTY","water_use","EMPIRICAL","ESTABLISHED",["agriculture","environment"])
    n_if_climate = mk_node("industrial_farming","CONTRIBUTES_TO","climate","EMPIRICAL","ESTABLISHED",["agriculture","climate"])

    mk_edge(n_ff_conf, n_ff_dist, "CAUSES", "Confinement prevents expression of natural behaviours, producing chronic distress", 0.9)
    mk_edge(n_ff_dist, n_ff_harm, "IMPLIES", "Distress in sentient creatures constitutes harm by any standard welfare definition", 0.9)
    mk_edge(n_pig_dist, n_ff_harm, "SUPPORTS", "Observed pig distress in farming conditions is direct evidence of harm", 0.85)
    mk_edge(n_if_ghg, n_if_climate, "CAUSES", "Greenhouse gases are the mechanism by which farming affects climate", 0.95)

    # ── Domain: Eating bacon / diet ethics ───────────────────────────────
    n_bacon_inst = mk_node("eating_bacon","IS_INSTANCE_OF","eating_meat","DEFINITIONAL","ESTABLISHED",["diet","food_system"])
    n_bacon_dep  = mk_node("eating_bacon","DEPENDS_ON","industrial_farming","EMPIRICAL","SUPPORTED",["diet","food_system"])
    n_meat_nutr  = mk_node("eating_meat","HAS_PROPERTY","nutritional_value","EMPIRICAL","SUPPORTED",["diet","nutrition"])
    n_meat_harm  = mk_node("eating_meat","CONTRIBUTES_TO","harm","LOGICAL","CONTESTED",["diet","ethics"])
    n_bacon_harm = mk_node("eating_bacon","CONTRIBUTES_TO","harm","LOGICAL","CONTESTED",["diet","ethics"])
    n_bacon_just = mk_node("eating_bacon","IS_JUSTIFIED_BY","necessity","VALUE","CONTESTED",["diet","ethics"])
    n_meat_nec   = mk_node("eating_meat","IS_NECESSARY_FOR","welfare","VALUE","CONTESTED",["diet","nutrition"])

    mk_edge(n_bacon_inst, n_bacon_dep, "REQUIRES", "Pork products at scale presuppose industrial pig production", 0.9)
    mk_edge(n_bacon_dep, n_bacon_harm, "CAUSES", "Consuming products that depend on harmful processes perpetuates those processes", 0.75)
    mk_edge(n_ff_harm, n_bacon_harm, "SUPPORTS", "The harm of factory farming is attributable (in part) to the consumer demand sustaining it", 0.7)
    mk_edge(n_meat_nutr, n_bacon_just, "SUPPORTS", "Genuine nutritional necessity could partially justify associated harms", 0.5)
    mk_edge(n_meat_nec, n_bacon_just, "SUPPORTS", "If meat is genuinely necessary for human welfare, this weighs in favour of its permissibility", 0.5)

    # ── Domain: Human diet morality (the 'pinned' topic) ─────────────────
    n_diet_moral = mk_node("eating_meat","VIOLATES","moral_framework","VALUE","CONTESTED",["diet","ethics","moral_philosophy"])
    n_human_auto = mk_node("human","HAS_PROPERTY","autonomy","VALUE","ESTABLISHED",["ethics","political_philosophy"])
    n_human_diet = mk_node("human","HAS_INTEREST_IN","welfare","VALUE","ESTABLISHED",["diet","ethics"])

    mk_edge(n_bacon_harm, n_diet_moral, "SUPPORTS", "If eating bacon causes harm and harm violates morality, bacon consumption is morally problematic", 0.8)
    mk_edge(n_animal_r, n_diet_moral, "SUPPORTS", "If animals have rights, practices that systematically violate them are morally wrong", 0.8)
    mk_edge(n_human_auto, n_bacon_just, "SUPPORTS", "Human autonomy over dietary choice is itself a competing moral value", 0.65)
    mk_edge(n_harm_moral, n_diet_moral, "IMPLIES", "If harm violates morality and eating meat causes harm, meat-eating violates morality", 0.8)

    # ── Domain: Utilitarian & deontological analysis ──────────────────────
    n_util_def   = mk_node("utilitarian_ethics","IS_DEFINED_BY","welfare","DEFINITIONAL","ESTABLISHED",["ethics","moral_philosophy"])
    n_util_eq    = mk_node("utilitarian_ethics","REQUIRES","equality","DEFINITIONAL","ESTABLISHED",["ethics","moral_philosophy"])
    n_util_harm  = mk_node("suffering","VIOLATES","utilitarian_ethics","LOGICAL","SUPPORTED",["ethics","moral_philosophy"])
    n_deon_def   = mk_node("deontological_ethics","IS_DEFINED_BY","rights","DEFINITIONAL","ESTABLISHED",["ethics","moral_philosophy"])
    n_virtue_def = mk_node("virtue_ethics","IS_DEFINED_BY","welfare","DEFINITIONAL","ESTABLISHED",["ethics","moral_philosophy"])

    mk_edge(n_util_harm, n_diet_moral, "SUPPORTS", "Under utilitarianism, systematically producing suffering is wrong regardless of who suffers", 0.85)
    mk_edge(n_deon_def, n_diet_moral, "SUPPORTS", "Under deontology, if animals have rights, violating them is categorically impermissible", 0.8)

    # ── Domain: Economic / systemic context ──────────────────────────────
    n_cap_if     = mk_node("capitalism","PRODUCES","industrial_farming","EMPIRICAL","SUPPORTED",["economics","agriculture"])
    n_cap_ineq   = mk_node("capitalism","CAUSES","inequality","EMPIRICAL","CONTESTED",["economics","political_philosophy"])
    n_reg_harm   = mk_node("regulation","PREVENTS","harm","LOGICAL","SUPPORTED",["policy","governance"])
    n_reg_cap    = mk_node("regulation","LIMITS","capitalism","EMPIRICAL","ESTABLISHED",["policy","economics"])
    n_corp_harm  = mk_node("corporation","CAUSES","harm","EMPIRICAL","SUPPORTED",["economics","ethics"])
    n_corp_env   = mk_node("corporation","CONTRIBUTES_TO","environmental_impact","EMPIRICAL","SUPPORTED",["economics","climate"])
    n_market_eff = mk_node("market","IS_PRECONDITION_FOR","abundance","LOGICAL","CONTESTED",["economics","political_philosophy"])
    n_market_ineq= mk_node("market","CONTRIBUTES_TO","inequality","EMPIRICAL","CONTESTED",["economics","political_philosophy"])

    mk_edge(n_cap_if, n_ff_harm, "CAUSES", "Capitalist incentive structures drive scale and intensity, amplifying industrial farming harm", 0.7)
    mk_edge(n_reg_harm, n_ff_harm, "PREVENTS", "Adequate regulation can constrain or eliminate factory farming practices that cause harm", 0.65)
    mk_edge(n_corp_harm, n_ff_harm, "SUPPORTS", "Corporate actors driving industrial farming are the proximate cause of its harms", 0.75)

    # ── Domain: Climate & environment ────────────────────────────────────
    n_climate_ext= mk_node("climate","CAUSES","extinction","EMPIRICAL","SUPPORTED",["climate","ecology"])
    n_if_eco     = mk_node("industrial_farming","UNDERMINES","ecosystem","EMPIRICAL","ESTABLISHED",["agriculture","ecology"])
    n_future_cli = mk_node("future_generation","IS_HARMED_BY","climate","LOGICAL","SUPPORTED",["ethics","climate"])
    n_future_eco = mk_node("future_generation","HAS_INTEREST_IN","ecosystem","VALUE","SUPPORTED",["ethics","ecology"])

    mk_edge(n_if_ghg, n_climate_ext, "CAUSES", "Farming-driven emissions accelerate climate change which drives species extinction", 0.75)
    mk_edge(n_climate_ext, n_future_eco, "UNDERMINES", "Extinction of species permanently destroys the ecosystems future generations depend on", 0.85)
    mk_edge(n_future_cli, n_diet_moral, "SUPPORTS", "Practices that harm future generations without their consent are morally problematic", 0.7)

    conn.commit()
    conn.close()

init_db()

# ── MODELS ────────────────────────────────────────────────────────────────────

class VocabEntityPropose(BaseModel):
    term: str
    category: str
    description: Optional[str] = None
    proposed_by: str = "anonymous"

class VocabPredicatePropose(BaseModel):
    term: str
    english: str
    direction_hint: str = "FORWARD"
    domain: Optional[str] = None
    range_hint: Optional[str] = None
    description: Optional[str] = None

class NodeCreate(BaseModel):
    subject: str
    predicate: str
    object: str
    node_type: str
    confidence: str = "SUPPORTED"
    domain_tags: List[str] = []
    created_by: str = "anonymous"

class EdgeCreate(BaseModel):
    source_id: str
    target_id: str
    relationship: str
    warrant_node_id: Optional[str] = None
    warrant_text: Optional[str] = None
    strength: float = 0.7
    conditions: Optional[str] = None
    created_by: str = "anonymous"

class ChallengeCreate(BaseModel):
    target_id: str
    target_type: str
    ground: str
    property_disputed: str
    argument: str
    counter_node_id: Optional[str] = None
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
    year: Optional[int] = None
    created_by: str = "anonymous"

class LLMProposal(BaseModel):
    nodes: List[NodeCreate] = []
    edges: List[EdgeCreate] = []
    evidence: List[EvidenceCreate] = []
    challenges: List[ChallengeCreate] = []
    proposed_by: str = "llm"
    reasoning: Optional[str] = None

class ViewCreate(BaseModel):
    name: str
    focus_node_id: Optional[str] = None
    pinned_node_ids: List[str] = []
    depth: int = 2
    created_by: str = "anonymous"

# ── FRONTEND ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r") as f:
        return f.read()

# ── VOCABULARY ────────────────────────────────────────────────────────────────

@app.get("/vocab", summary="Get all vocabulary (entities + predicates)")
def get_vocab():
    conn = get_db()
    entities = [row_to_dict(r) for r in conn.execute(
        "SELECT * FROM vocab_entities WHERE status='ACTIVE' ORDER BY category, term"
    ).fetchall()]
    predicates = [row_to_dict(r) for r in conn.execute(
        "SELECT * FROM vocab_predicates WHERE status='ACTIVE' ORDER BY term"
    ).fetchall()]
    conn.close()
    return {"entities": entities, "predicates": predicates}

@app.post("/vocab/entities", summary="Add a new entity term to the vocabulary")
def add_entity(data: VocabEntityPropose):
    conn = get_db()
    existing = conn.execute("SELECT id FROM vocab_entities WHERE term=?", (data.term,)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(409, f"Entity '{data.term}' already exists")
    valid_cats = ["AGENT","CONCEPT","PROCESS","STATE","PROPERTY","SYSTEM"]
    if data.category not in valid_cats:
        raise HTTPException(422, f"Category must be one of {valid_cats}")
    eid = new_id()
    term = data.term.lower().replace(" ","_")
    conn.execute(
        "INSERT INTO vocab_entities VALUES (?,?,?,?,?,?,?)",
        (eid, term, data.category, data.description, data.proposed_by, "ACTIVE", now())
    )
    conn.commit()
    row = conn.execute("SELECT * FROM vocab_entities WHERE id=?", (eid,)).fetchone()
    conn.close()
    return row_to_dict(row)

@app.post("/vocab/predicates", summary="Add a new predicate to the vocabulary")
def add_predicate(data: VocabPredicatePropose):
    conn = get_db()
    existing = conn.execute("SELECT id FROM vocab_predicates WHERE term=?", (data.term,)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(409, f"Predicate '{data.term}' already exists")
    pid = new_id()
    conn.execute(
        "INSERT INTO vocab_predicates VALUES (?,?,?,?,?,?,?,?,?)",
        (pid, data.term.upper(), data.english, data.direction_hint,
         data.domain, data.range_hint, data.description, "ACTIVE", now())
    )
    conn.commit()
    row = conn.execute("SELECT * FROM vocab_predicates WHERE id=?", (pid,)).fetchone()
    conn.close()
    return row_to_dict(row)

# ── NODES ─────────────────────────────────────────────────────────────────────

@app.get("/nodes")
def list_nodes(
    tag: Optional[str] = None,
    search: Optional[str] = None,
    confidence: Optional[str] = None,
    node_type: Optional[str] = None
):
    conn = get_db()
    q = "SELECT * FROM nodes WHERE is_deleted=0"
    params = []
    if tag:
        q += " AND domain_tags LIKE ?"
        params.append(f'%"{tag}"%')
    if search:
        q += " AND statement LIKE ?"
        params.append(f"%{search}%")
    if confidence:
        q += " AND confidence=?"
        params.append(confidence)
    if node_type:
        q += " AND node_type=?"
        params.append(node_type)
    q += " ORDER BY created_at DESC"
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]

@app.post("/nodes")
def create_node(data: NodeCreate):
    conn = get_db()
    subj = conn.execute("SELECT id FROM vocab_entities WHERE term=? AND status='ACTIVE'", (data.subject,)).fetchone()
    pred = conn.execute("SELECT id, english FROM vocab_predicates WHERE term=? AND status='ACTIVE'", (data.predicate,)).fetchone()
    obj  = conn.execute("SELECT id FROM vocab_entities WHERE term=? AND status='ACTIVE'", (data.object,)).fetchone()
    if not subj:
        raise HTTPException(422, f"Unknown entity: '{data.subject}'. Add via POST /vocab/entities first.")
    if not pred:
        raise HTTPException(422, f"Unknown predicate: '{data.predicate}'. Add via POST /vocab/predicates first.")
    if not obj:
        raise HTTPException(422, f"Unknown entity: '{data.object}'. Add via POST /vocab/entities first.")
    valid_types = ["EMPIRICAL","LOGICAL","VALUE","DEFINITIONAL","OBSERVED"]
    if data.node_type not in valid_types:
        raise HTTPException(422, f"node_type must be one of {valid_types}")
    statement = f"{data.subject} {pred['english']} {data.object}".replace("_", " ")
    nid = new_id()
    conn.execute(
        "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (nid, subj["id"], pred["id"], obj["id"], statement,
         data.node_type, data.confidence, json.dumps(data.domain_tags),
         data.created_by, now(), 0, None)
    )
    conn.commit()
    row = conn.execute("SELECT * FROM nodes WHERE id=?", (nid,)).fetchone()
    conn.close()
    return row_to_dict(row)

@app.get("/nodes/{node_id}")
def get_node(node_id: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Node not found")
    node = row_to_dict(row)
    node["outgoing_edges"] = [row_to_dict(r) for r in conn.execute(
        "SELECT * FROM edges WHERE source_id=? AND is_deleted=0", (node_id,)).fetchall()]
    node["incoming_edges"] = [row_to_dict(r) for r in conn.execute(
        "SELECT * FROM edges WHERE target_id=? AND is_deleted=0", (node_id,)).fetchall()]
    node["evidence"] = [row_to_dict(r) for r in conn.execute(
        "SELECT * FROM evidence WHERE target_id=? AND target_type='NODE'", (node_id,)).fetchall()]
    node["challenges"] = [row_to_dict(r) for r in conn.execute(
        "SELECT * FROM challenges WHERE target_id=? AND target_type='NODE'", (node_id,)).fetchall()]
    conn.close()
    return node

@app.delete("/nodes/{node_id}")
def delete_node(node_id: str):
    conn = get_db()
    conn.execute("UPDATE nodes SET is_deleted=1 WHERE id=?", (node_id,))
    conn.commit()
    conn.close()
    return {"deleted": True}

# ── EDGES ─────────────────────────────────────────────────────────────────────

@app.get("/edges")
def list_edges():
    conn = get_db()
    rows = conn.execute("SELECT * FROM edges WHERE is_deleted=0").fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]

@app.post("/edges")
def create_edge(data: EdgeCreate):
    conn = get_db()
    if not conn.execute("SELECT id FROM nodes WHERE id=? AND is_deleted=0", (data.source_id,)).fetchone():
        raise HTTPException(404, f"Source node not found")
    if not conn.execute("SELECT id FROM nodes WHERE id=? AND is_deleted=0", (data.target_id,)).fetchone():
        raise HTTPException(404, f"Target node not found")
    if not data.warrant_node_id and not data.warrant_text:
        raise HTTPException(422, "Edge requires warrant_node_id or warrant_text")
    valid_rels = ["SUPPORTS","UNDERMINES","REQUIRES","CONTRADICTS","CAUSES","IMPLIES",
                  "CORRELATES_WITH","DEFINES","EXEMPLIFIES","PREVENTS"]
    if data.relationship not in valid_rels:
        raise HTTPException(422, f"relationship must be one of {valid_rels}")
    eid = new_id()
    conn.execute(
        "INSERT INTO edges VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (eid, data.source_id, data.target_id, data.relationship,
         data.warrant_node_id, data.warrant_text, data.strength,
         data.conditions, data.created_by, now(), 0)
    )
    conn.commit()
    row = conn.execute("SELECT * FROM edges WHERE id=?", (eid,)).fetchone()
    conn.close()
    return row_to_dict(row)

@app.delete("/edges/{edge_id}")
def delete_edge(edge_id: str):
    conn = get_db()
    conn.execute("UPDATE edges SET is_deleted=1 WHERE id=?", (edge_id,))
    conn.commit()
    conn.close()
    return {"deleted": True}

# ── CHALLENGES ────────────────────────────────────────────────────────────────

@app.post("/challenges")
def create_challenge(data: ChallengeCreate):
    conn = get_db()
    cid = new_id()
    conn.execute(
        "INSERT INTO challenges VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (cid, data.target_id, data.target_type, data.ground,
         data.property_disputed, data.argument, data.counter_node_id,
         "OPEN", None, data.created_by, now())
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

# ── EVIDENCE ──────────────────────────────────────────────────────────────────

@app.post("/evidence")
def create_evidence(data: EvidenceCreate):
    conn = get_db()
    eid = new_id()
    conn.execute(
        "INSERT INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?)",
        (eid, data.target_id, data.target_type, data.source_type,
         data.description, data.url, data.citation, data.year,
         data.created_by, now())
    )
    conn.commit()
    row = conn.execute("SELECT * FROM evidence WHERE id=?", (eid,)).fetchone()
    conn.close()
    return row_to_dict(row)

# ── GRAPH & PATH ──────────────────────────────────────────────────────────────

@app.get("/graph")
def get_graph(tag: Optional[str] = None):
    conn = get_db()
    if tag:
        nodes = [row_to_dict(r) for r in conn.execute(
            "SELECT * FROM nodes WHERE is_deleted=0 AND domain_tags LIKE ?",
            (f'%"{tag}"%',)).fetchall()]
        node_ids = {n["id"] for n in nodes}
        all_edges = [row_to_dict(r) for r in conn.execute("SELECT * FROM edges WHERE is_deleted=0").fetchall()]
        edges = [e for e in all_edges if e["source_id"] in node_ids or e["target_id"] in node_ids]
    else:
        nodes = [row_to_dict(r) for r in conn.execute("SELECT * FROM nodes WHERE is_deleted=0").fetchall()]
        edges = [row_to_dict(r) for r in conn.execute("SELECT * FROM edges WHERE is_deleted=0").fetchall()]
    for n in nodes:
        n["open_challenges"] = conn.execute(
            "SELECT COUNT(*) FROM challenges WHERE target_id=? AND status='OPEN'", (n["id"],)
        ).fetchone()[0]
        n["domain_tags"] = json.loads(n["domain_tags"])
    conn.close()
    return {"nodes": nodes, "edges": edges}

@app.get("/view", summary="Focused view: local neighbourhood + paths to pinned nodes")
def get_view(
    focus_id: str,
    pinned: Optional[str] = Query(None, description="Comma-separated node IDs to keep visible"),
    depth: int = Query(2, description="Hops from focus node")
):
    conn = get_db()
    pinned_ids = [p.strip() for p in pinned.split(",")] if pinned else []

    def get_neighbors(node_id, hops):
        visited = {node_id}
        frontier = {node_id}
        for _ in range(hops):
            next_f = set()
            for nid in frontier:
                for e in conn.execute(
                    "SELECT source_id, target_id FROM edges WHERE (source_id=? OR target_id=?) AND is_deleted=0",
                    (nid, nid)).fetchall():
                    for cand in [e["source_id"], e["target_id"]]:
                        if cand not in visited:
                            next_f.add(cand)
                            visited.add(cand)
            frontier = next_f
        return visited

    def find_path(start, end):
        if start == end:
            return [start]
        queue = [[start]]
        visited = {start}
        while queue:
            path = queue.pop(0)
            current = path[-1]
            for e in conn.execute(
                "SELECT source_id, target_id FROM edges WHERE (source_id=? OR target_id=?) AND is_deleted=0",
                (current, current)).fetchall():
                for nb in [e["source_id"], e["target_id"]]:
                    if nb == end:
                        return path + [nb]
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(path + [nb])
        return []

    local_ids = get_neighbors(focus_id, depth)
    path_ids = set()
    paths = {}
    for pid in pinned_ids:
        path = find_path(focus_id, pid)
        if path:
            paths[pid] = path
            path_ids.update(path)

    all_ids = local_ids | path_ids | set(pinned_ids) | {focus_id}
    placeholders = ",".join("?" * len(all_ids))
    nodes = [row_to_dict(r) for r in conn.execute(
        f"SELECT * FROM nodes WHERE id IN ({placeholders}) AND is_deleted=0",
        list(all_ids)).fetchall()]
    node_id_set = {n["id"] for n in nodes}
    all_edges = [row_to_dict(r) for r in conn.execute("SELECT * FROM edges WHERE is_deleted=0").fetchall()]
    edges = [e for e in all_edges if e["source_id"] in node_id_set and e["target_id"] in node_id_set]

    for n in nodes:
        n["domain_tags"] = json.loads(n["domain_tags"])
        n["is_focus"] = n["id"] == focus_id
        n["is_pinned"] = n["id"] in pinned_ids
        n["is_path"] = n["id"] in path_ids

    conn.close()
    return {"focus_id": focus_id, "pinned_ids": pinned_ids, "paths": paths, "nodes": nodes, "edges": edges}

@app.get("/path", summary="Find shortest reasoning path between two nodes")
def find_path_endpoint(from_id: str, to_id: str):
    conn = get_db()
    if from_id == to_id:
        conn.close()
        return {"path": [from_id], "nodes": [], "edges": []}
    queue = [[from_id]]
    visited = {from_id}
    found = []
    while queue:
        path = queue.pop(0)
        current = path[-1]
        for e in conn.execute(
            "SELECT source_id, target_id FROM edges WHERE (source_id=? OR target_id=?) AND is_deleted=0",
            (current, current)).fetchall():
            for nb in [e["source_id"], e["target_id"]]:
                if nb == to_id:
                    found = path + [nb]
                    break
                if nb not in visited:
                    visited.add(nb)
                    queue.append(path + [nb])
            if found:
                break
        if found:
            break

    if not found:
        conn.close()
        return {"path": [], "nodes": [], "edges": []}

    placeholders = ",".join("?" * len(found))
    nodes = [row_to_dict(r) for r in conn.execute(
        f"SELECT * FROM nodes WHERE id IN ({placeholders})", found).fetchall()]
    path_set = set(found)
    all_edges = [row_to_dict(r) for r in conn.execute("SELECT * FROM edges WHERE is_deleted=0").fetchall()]
    edges = [e for e in all_edges if e["source_id"] in path_set and e["target_id"] in path_set]
    conn.close()
    return {"path": found, "nodes": nodes, "edges": edges}

# ── LLM ENDPOINTS ─────────────────────────────────────────────────────────────

@app.get("/llm/context", summary="[LLM] Full orientation: vocab + stats + instructions")
def llm_context():
    conn = get_db()
    entities = [row_to_dict(r) for r in conn.execute(
        "SELECT term, category, description FROM vocab_entities WHERE status='ACTIVE' ORDER BY category, term"
    ).fetchall()]
    predicates = [row_to_dict(r) for r in conn.execute(
        "SELECT term, english, direction_hint, description FROM vocab_predicates WHERE status='ACTIVE'"
    ).fetchall()]
    node_count = conn.execute("SELECT COUNT(*) FROM nodes WHERE is_deleted=0").fetchone()[0]
    edge_count = conn.execute("SELECT COUNT(*) FROM edges WHERE is_deleted=0").fetchone()[0]
    contested = [row_to_dict(r) for r in conn.execute(
        "SELECT id, statement, confidence, domain_tags FROM nodes WHERE confidence IN ('CONTESTED','SPECULATIVE') AND is_deleted=0"
    ).fetchall()]
    open_challenges = conn.execute("SELECT COUNT(*) FROM challenges WHERE status='OPEN'").fetchone()[0]
    domain_counts = {}
    for row in conn.execute("SELECT domain_tags FROM nodes WHERE is_deleted=0").fetchall():
        try:
            for tag in json.loads(row[0]):
                domain_counts[tag] = domain_counts.get(tag, 0) + 1
        except Exception:
            pass
    conn.close()
    return {
        "description": "Logicpedia: a universal structured reasoning graph. Nodes are typed triples [subject][predicate][object]. All terms must come from the vocabulary.",
        "stats": {"nodes": node_count, "edges": edge_count, "open_challenges": open_challenges},
        "domains": domain_counts,
        "contested_nodes": contested,
        "vocabulary": {"entities": entities, "predicates": predicates},
        "node_types": ["EMPIRICAL","LOGICAL","VALUE","DEFINITIONAL","OBSERVED"],
        "confidence_levels": ["ESTABLISHED","SUPPORTED","CONTESTED","SPECULATIVE","REFUTED"],
        "edge_relationships": ["SUPPORTS","UNDERMINES","REQUIRES","CONTRADICTS","CAUSES","IMPLIES","CORRELATES_WITH","DEFINES","EXEMPLIFIES","PREVENTS"],
        "instructions": {
            "orient": "Call GET /llm/context first to see all vocabulary",
            "add_node": "POST /nodes — {subject, predicate, object, node_type, confidence, domain_tags}. All terms must exist in vocab.",
            "add_vocab": "POST /vocab/entities or /vocab/predicates to add new terms before using them in nodes",
            "add_edge": "POST /edges — {source_id, target_id, relationship, warrant_text, strength}",
            "bulk_add": "POST /llm/propose — {nodes[], edges[], evidence[], challenges[], reasoning}",
            "search": "GET /nodes?search=term or GET /nodes?tag=domain",
            "path": "GET /path?from_id=X&to_id=Y",
            "view": "GET /view?focus_id=X&pinned=Y,Z&depth=2"
        }
    }

@app.post("/llm/propose", summary="[LLM] Batch-add nodes, edges, evidence and challenges")
def llm_propose(data: LLMProposal):
    results = {
        "reasoning": data.reasoning,
        "proposed_by": data.proposed_by,
        "created": {"nodes": [], "edges": [], "evidence": [], "challenges": []},
        "errors": []
    }
    conn = get_db()
    node_id_map = {}  # statement → id for cross-referencing within same proposal

    for nd in data.nodes:
        try:
            subj = conn.execute("SELECT id FROM vocab_entities WHERE term=? AND status='ACTIVE'", (nd.subject,)).fetchone()
            pred = conn.execute("SELECT id, english FROM vocab_predicates WHERE term=? AND status='ACTIVE'", (nd.predicate,)).fetchone()
            obj  = conn.execute("SELECT id FROM vocab_entities WHERE term=? AND status='ACTIVE'", (nd.object,)).fetchone()
            if not subj or not pred or not obj:
                results["errors"].append({"item": f"node:{nd.subject}/{nd.predicate}/{nd.object}", "error": "Unknown vocab term — add to /vocab/entities or /vocab/predicates first"})
                continue
            statement = f"{nd.subject} {pred['english']} {nd.object}".replace("_", " ")
            nid = new_id()
            conn.execute(
                "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (nid, subj["id"], pred["id"], obj["id"], statement,
                 nd.node_type, nd.confidence, json.dumps(nd.domain_tags),
                 data.proposed_by, now(), 0, None)
            )
            results["created"]["nodes"].append({"id": nid, "statement": statement})
            node_id_map[statement] = nid
        except Exception as ex:
            results["errors"].append({"item": f"node:{nd.subject}/{nd.predicate}/{nd.object}", "error": str(ex)})

    for ed in data.edges:
        try:
            if not conn.execute("SELECT id FROM nodes WHERE id=? AND is_deleted=0", (ed.source_id,)).fetchone():
                results["errors"].append({"item": f"edge:{ed.source_id}→{ed.target_id}", "error": "Source node not found"})
                continue
            if not conn.execute("SELECT id FROM nodes WHERE id=? AND is_deleted=0", (ed.target_id,)).fetchone():
                results["errors"].append({"item": f"edge:{ed.source_id}→{ed.target_id}", "error": "Target node not found"})
                continue
            eid = new_id()
            conn.execute(
                "INSERT INTO edges VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (eid, ed.source_id, ed.target_id, ed.relationship,
                 ed.warrant_node_id, ed.warrant_text, ed.strength,
                 ed.conditions, data.proposed_by, now(), 0)
            )
            results["created"]["edges"].append({"id": eid})
        except Exception as ex:
            results["errors"].append({"item": f"edge:{ed.source_id}→{ed.target_id}", "error": str(ex)})

    for ev in data.evidence:
        try:
            eid = new_id()
            conn.execute(
                "INSERT INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?)",
                (eid, ev.target_id, ev.target_type, ev.source_type,
                 ev.description, ev.url, ev.citation, ev.year,
                 data.proposed_by, now())
            )
            results["created"]["evidence"].append({"id": eid})
        except Exception as ex:
            results["errors"].append({"item": f"evidence:{ev.target_id}", "error": str(ex)})

    for ch in data.challenges:
        try:
            cid = new_id()
            conn.execute(
                "INSERT INTO challenges VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (cid, ch.target_id, ch.target_type, ch.ground,
                 ch.property_disputed, ch.argument, ch.counter_node_id,
                 "OPEN", None, data.proposed_by, now())
            )
            results["created"]["challenges"].append({"id": cid})
        except Exception as ex:
            results["errors"].append({"item": f"challenge:{ch.target_id}", "error": str(ex)})

    conn.commit()
    conn.close()
    return results

# ── VIEWS ─────────────────────────────────────────────────────────────────────

@app.post("/views")
def save_view(data: ViewCreate):
    conn = get_db()
    vid = new_id()
    conn.execute(
        "INSERT INTO views VALUES (?,?,?,?,?,?,?)",
        (vid, data.name, data.focus_node_id, json.dumps(data.pinned_node_ids),
         data.depth, data.created_by, now())
    )
    conn.commit()
    row = conn.execute("SELECT * FROM views WHERE id=?", (vid,)).fetchone()
    conn.close()
    return row_to_dict(row)

@app.get("/views")
def list_views():
    conn = get_db()
    rows = conn.execute("SELECT * FROM views ORDER BY created_at DESC").fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]
