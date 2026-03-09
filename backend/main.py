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
            superseded_by TEXT,
            is_featured INTEGER NOT NULL DEFAULT 0
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
    # Migration: add is_featured if not present
    try:
        conn.execute("ALTER TABLE nodes ADD COLUMN is_featured INTEGER NOT NULL DEFAULT 0")
        conn.commit()
    except Exception:
        pass
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

    def mk_node(subj, pred, obj, ntype, conf, tags, by="seed", featured=0):
        pid, peng = p(pred)
        statement = f"{subj} {peng} {obj}".replace("_", " ")
        nid = new_id()
        conn.execute(
            "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (nid, e(subj), pid, e(obj), statement, ntype, conf, json.dumps(tags), by, now(), 0, None, featured)
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


    # ── Domain: Foundational moral axioms (FEATURED) ─────────────────────
    # These are the high-level nodes that ground the entire graph.
    # Add extra vocab needed first
    extra_entities = [
        ("happiness", "CONCEPT", "Positive subjective experience and flourishing"),
        ("well_being", "CONCEPT", "Overall positive state of an entity across all dimensions"),
        ("moral_agent", "AGENT", "Any being capable of moral reasoning and responsibility"),
        ("sentient_being", "AGENT", "Any being capable of subjective experience including pleasure and pain"),
        ("interest", "CONCEPT", "A stake in outcomes that matters to the holder"),
        ("equal_consideration", "CONCEPT", "The principle that equal interests deserve equal weight regardless of who holds them"),
        ("unnecessary_suffering", "CONCEPT", "Suffering that serves no sufficiently weighty purpose"),
        ("golden_rule", "CONCEPT", "The principle of treating others as one would wish to be treated"),
        ("moral_circle", "CONCEPT", "The set of beings whose interests are granted moral consideration"),
        ("flourishing", "CONCEPT", "The fullest expression of a being's capacities and positive states"),
        ("preference", "CONCEPT", "A ranking of outcomes by a being that expresses what it wants"),
        ("impartiality", "CONCEPT", "Judging without favouring one party based on identity rather than interests"),
    ]
    extra_predicates = [
        ("DEMANDS", "demands", "FORWARD", None, None, "Subject requires or necessitates object as a moral or logical consequence"),
        ("GROUNDS", "grounds", "FORWARD", None, None, "Subject provides the foundational basis for object"),
        ("EXTENDS_TO", "extends to", "FORWARD", None, None, "Subject applies or reaches to include object"),
        ("OBLIGATES", "obligates", "FORWARD", None, None, "Subject creates a duty or obligation toward object"),
        ("DIMINISHES", "diminishes", "FORWARD", None, None, "Subject reduces or weakens object"),
        ("DESERVES", "deserves", "FORWARD", "AGENT", None, "Subject is entitled to or merits object"),
        ("MAXIMISES", "maximises", "FORWARD", None, None, "Subject increases object to the greatest possible degree"),
        ("MINIMISES", "minimises", "FORWARD", None, None, "Subject reduces object to the least possible degree"),
    ]
    t2 = now()
    for term, cat, desc in extra_entities:
        existing = conn.execute("SELECT id FROM vocab_entities WHERE term=?", (term,)).fetchone()
        if not existing:
            conn.execute("INSERT INTO vocab_entities VALUES (?,?,?,?,?,?,?)",
                (new_id(), term, cat, desc, 'system', 'ACTIVE', t2))
    for term, eng, dirn, dom, rng, desc in extra_predicates:
        existing = conn.execute("SELECT id FROM vocab_predicates WHERE term=?", (term,)).fetchone()
        if not existing:
            conn.execute("INSERT INTO vocab_predicates VALUES (?,?,?,?,?,?,?,?,?)",
                (new_id(), term, eng, dirn, dom, rng, desc, 'ACTIVE', t2))
    conn.commit()

    # Now seed the featured axiom nodes
    ax_suffering_bad   = mk_node("suffering","IS_SUFFICIENT_FOR","moral_worth","VALUE","ESTABLISHED",["ethics","moral_philosophy","axiomatic"], featured=1)
    ax_equal_interest  = mk_node("equal_consideration","GROUNDS","moral_framework","VALUE","SUPPORTED",["ethics","moral_philosophy","axiomatic"], featured=1)
    ax_sentient_circle = mk_node("moral_circle","EXTENDS_TO","sentient_being","VALUE","CONTESTED",["ethics","moral_philosophy","axiomatic"], featured=1)
    ax_harm_obligates  = mk_node("unnecessary_suffering","OBLIGATES","moral_agent","VALUE","SUPPORTED",["ethics","moral_philosophy","axiomatic"], featured=1)
    ax_flourish        = mk_node("sentient_being","DESERVES","flourishing","VALUE","SUPPORTED",["ethics","moral_philosophy","axiomatic"], featured=1)
    ax_impartial       = mk_node("impartiality","GROUNDS","moral_framework","VALUE","SUPPORTED",["ethics","moral_philosophy","axiomatic"], featured=1)
    ax_interest_equal  = mk_node("interest","DEMANDS","equal_consideration","VALUE","SUPPORTED",["ethics","moral_philosophy","axiomatic"], featured=1)
    ax_welfare_max     = mk_node("well_being","IS_SUFFICIENT_FOR","moral_worth","VALUE","SUPPORTED",["ethics","moral_philosophy","axiomatic"], featured=1)
    ax_golden          = mk_node("golden_rule","GROUNDS","moral_framework","VALUE","ESTABLISHED",["ethics","moral_philosophy","axiomatic"], featured=1)
    ax_pref_respect    = mk_node("preference","DEMANDS","equal_consideration","VALUE","SUPPORTED",["ethics","moral_philosophy","axiomatic"], featured=1)
    ax_consciousness   = mk_node("consciousness","IS_SUFFICIENT_FOR","moral_worth","VALUE","CONTESTED",["ethics","moral_philosophy","axiomatic"], featured=1)
    ax_harm_bad        = mk_node("harm","VIOLATES","moral_framework","VALUE","ESTABLISHED",["ethics","moral_philosophy","axiomatic"], featured=1)

    # Connect axioms to each other and to the rest of the graph
    mk_edge(ax_suffering_bad, ax_sentient_circle, "IMPLIES",
        "If suffering alone grounds moral worth, then all sentient beings — capable of suffering — fall within the moral circle", 0.9)
    mk_edge(ax_equal_interest, ax_impartial, "REQUIRES",
        "Equal consideration of interests requires the impartial standpoint — judging by what interests are at stake, not who holds them", 0.9)
    mk_edge(ax_interest_equal, ax_equal_interest, "IMPLIES",
        "The claim that interests demand equal consideration is just what equal consideration means", 0.95)
    mk_edge(ax_sentient_circle, ax_harm_obligates, "IMPLIES",
        "Once sentient beings are in the moral circle, their suffering creates obligations to avoid causing it unnecessarily", 0.9)
    mk_edge(ax_golden, ax_impartial, "SUPPORTS",
        "The golden rule is an early expression of impartiality — it asks us to weight others' experiences as we weight our own", 0.8)
    mk_edge(ax_flourish, ax_welfare_max, "SUPPORTS",
        "If beings deserve to flourish, then well-being — the metric of flourishing — has moral weight", 0.85)
    mk_edge(ax_pref_respect, ax_equal_interest, "IMPLIES",
        "If preferences demand equal consideration, this just is the claim that equal interests (expressed as preferences) matter equally", 0.9)
    mk_edge(ax_consciousness, ax_suffering_bad, "SUPPORTS",
        "Consciousness is the precondition for suffering — without it, there is no one to suffer", 0.95)
    mk_edge(ax_harm_bad, ax_harm_obligates, "IMPLIES",
        "If harm violates morality, then agents capable of preventing unnecessary harm are obligated to do so", 0.9)
    mk_edge(ax_impartial, n_harm_moral, "SUPPORTS",
        "The impartiality principle implies harm to any being deserves equal weight regardless of species", 0.8)
    mk_edge(ax_suffering_bad, n_sent_mw, "SUPPORTS",
        "The foundational axiom that suffering grounds moral worth directly supports the contested claim about sentient creatures", 0.9)
    mk_edge(ax_equal_interest, n_sent_int, "IMPLIES",
        "Equal consideration of interests implies that sentient creatures' welfare interests cannot simply be dismissed", 0.85)
    mk_edge(ax_harm_obligates, n_ff_harm, "SUPPORTS",
        "The obligation to avoid unnecessary suffering connects directly to factory farming as an unnecessary source of mass suffering", 0.85)

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

@app.get("/nodes/featured", summary="Featured axiom nodes for the home screen")
def get_featured_nodes():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM nodes WHERE is_featured=1 AND is_deleted=0 ORDER BY created_at ASC"
    ).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]

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
        "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (nid, subj["id"], pred["id"], obj["id"], statement,
         data.node_type, data.confidence, json.dumps(data.domain_tags),
         data.created_by, now(), 0, None, 0)
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
    # Grab real node IDs for examples
    sample_nodes = conn.execute(
        "SELECT id, statement FROM nodes WHERE is_deleted=0 LIMIT 2"
    ).fetchall()
    ex_node_id_1 = sample_nodes[0]["id"] if len(sample_nodes) > 0 else "UUID-OF-NODE-1"
    ex_node_id_2 = sample_nodes[1]["id"] if len(sample_nodes) > 1 else "UUID-OF-NODE-2"
    ex_subj = entities[0]["term"] if entities else "pig"
    ex_pred = predicates[0]["term"] if predicates else "HAS_PROPERTY"
    ex_obj  = entities[1]["term"] if len(entities) > 1 else "suffering"
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
        "source_types": ["PEER_REVIEWED","META_ANALYSIS","SYSTEMATIC_REVIEW","OBSERVATIONAL","EXPERT_CONSENSUS","LOGICAL_DERIVATION","ANECDOTAL"],
        "challenge_grounds": ["EMPIRICAL","LOGICAL","VALUE","SCOPE","DEFINITION"],
        "instructions": {
            "step_1": "GET /llm/context — read vocabulary, domains, contested nodes, and examples",
            "step_2_vocab": "If you need a new term: POST /vocab/entities {term, category, description} or POST /vocab/predicates {term, english, description}",
            "step_3_nodes": "POST /nodes — {subject, predicate, object, node_type, confidence, domain_tags}. All three terms must exist in vocabulary.",
            "step_4_edges": "POST /edges — {source_id, target_id, relationship, warrant_text, strength}. source_id and target_id are UUID strings from node objects.",
            "step_5_evidence": "POST /evidence — {target_id, target_type, source_type, description, url, citation, year}",
            "step_6_challenge": "POST /challenges — {target_id, target_type, ground, property_disputed, argument}",
            "bulk_alternative": "POST /llm/propose — do steps 3-6 in one call. Edges still require UUID source_id/target_id.",
            "curl_guide": "GET /llm/curl-guide — get copy-paste curl commands with real IDs pre-filled from the live graph",
            "critical_warning": "NEVER use statement text in place of UUIDs. source_id, target_id, target_id in edges/evidence/challenges must all be UUID strings like 'a1b2c3d4-...'"
        },
        "examples": {
            "add_entity_vocab": {
                "POST /vocab/entities": {
                    "term": "epistemic_injustice",
                    "category": "CONCEPT",
                    "description": "Harm done to someone in their capacity as a knower"
                }
            },
            "add_predicate_vocab": {
                "POST /vocab/predicates": {
                    "term": "PRESUPPOSES",
                    "english": "presupposes",
                    "description": "Subject cannot be meaningfully asserted without assuming object"
                }
            },
            "add_node": {
                "POST /nodes": {
                    "subject": ex_subj,
                    "predicate": ex_pred,
                    "object": ex_obj,
                    "node_type": "EMPIRICAL",
                    "confidence": "SUPPORTED",
                    "domain_tags": ["example_domain"]
                }
            },
            "add_edge": {
                "POST /edges": {
                    "source_id": ex_node_id_1,
                    "target_id": ex_node_id_2,
                    "relationship": "SUPPORTS",
                    "warrant_text": "The source claim provides empirical grounding for the target claim",
                    "strength": 0.8
                },
                "note": f"These are real node IDs from the live graph. Replace with the IDs of the nodes you want to connect."
            },
            "add_evidence": {
                "POST /evidence": {
                    "target_id": ex_node_id_1,
                    "target_type": "NODE",
                    "source_type": "PEER_REVIEWED",
                    "description": "Brief description of what the source shows",
                    "url": "https://doi.org/example",
                    "citation": "Author et al. (2023). Journal Name, Vol(Issue).",
                    "year": 2023
                }
            },
            "add_challenge": {
                "POST /challenges": {
                    "target_id": ex_node_id_1,
                    "target_type": "NODE",
                    "ground": "EMPIRICAL",
                    "property_disputed": "confidence",
                    "argument": "The evidence for this claim does not establish causation, only correlation",
                    "counter_node_id": None
                }
            },
            "bulk_propose": {
                "POST /llm/propose": {
                    "proposed_by": "your-llm-name",
                    "reasoning": "Explain what you are adding and why",
                    "nodes": [
                        {
                            "subject": ex_subj,
                            "predicate": ex_pred,
                            "object": ex_obj,
                            "node_type": "LOGICAL",
                            "confidence": "SUPPORTED",
                            "domain_tags": ["example"]
                        }
                    ],
                    "edges": [
                        {
                            "source_id": ex_node_id_1,
                            "target_id": ex_node_id_2,
                            "relationship": "SUPPORTS",
                            "warrant_text": "Warrant explaining the connection",
                            "strength": 0.75
                        }
                    ],
                    "evidence": [],
                    "challenges": []
                },
                "critical_note": "Edges require UUID source_id/target_id — NOT statement text. Get IDs from GET /nodes or from the created.nodes array returned by this call."
            }
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
                "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (nid, subj["id"], pred["id"], obj["id"], statement,
                 nd.node_type, nd.confidence, json.dumps(nd.domain_tags),
                 data.proposed_by, now(), 0, None, 0)
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

# ── CURL GUIDE ────────────────────────────────────────────────────────────────

@app.get("/llm/curl-guide", summary="[LLM] Ready-to-run curl commands with real IDs from the live graph")
def llm_curl_guide():
    BASE = "https://logicpedia-production.up.railway.app"
    conn = get_db()

    entities = [row_to_dict(r) for r in conn.execute(
        "SELECT term, category FROM vocab_entities WHERE status='ACTIVE' ORDER BY category, term"
    ).fetchall()]
    predicates = [row_to_dict(r) for r in conn.execute(
        "SELECT term, english FROM vocab_predicates WHERE status='ACTIVE'"
    ).fetchall()]
    nodes_sample = [row_to_dict(r) for r in conn.execute(
        "SELECT id, statement, node_type, confidence, domain_tags FROM nodes WHERE is_deleted=0 ORDER BY created_at LIMIT 20"
    ).fetchall()]
    for n in nodes_sample:
        n["domain_tags"] = json.loads(n["domain_tags"])
    contested = [row_to_dict(r) for r in conn.execute(
        "SELECT id, statement, confidence FROM nodes WHERE confidence IN ('CONTESTED','SPECULATIVE') AND is_deleted=0 LIMIT 10"
    ).fetchall()]

    ex_subj = entities[0]["term"] if entities else "pig"
    ex_pred = predicates[0]["term"] if predicates else "HAS_PROPERTY"
    ex_obj  = entities[1]["term"] if len(entities) > 1 else "suffering"
    ex_id_1 = nodes_sample[0]["id"] if nodes_sample else "REPLACE-WITH-NODE-UUID"
    ex_id_2 = nodes_sample[1]["id"] if len(nodes_sample) > 1 else "REPLACE-WITH-NODE-UUID"

    conn.close()

    return {
        "intro": (
            "Ready-to-run curl commands pre-filled with real vocab terms and node IDs from the live graph. "
            "Save each file_contents block to the named file, then run the curl command."
        ),
        "base_url": BASE,
        "steps": {
            "step_0_explore": {
                "description": "Browse the graph before adding anything",
                "commands": [
                    f"curl '{BASE}/graph' | python3 -m json.tool | head -100",
                    f"curl '{BASE}/nodes?tag=ethics' | python3 -m json.tool",
                    f"curl '{BASE}/nodes?search=suffering' | python3 -m json.tool",
                    f"curl '{BASE}/nodes?confidence=CONTESTED' | python3 -m json.tool",
                ]
            },
            "step_1_add_vocab_entity": {
                "description": "Add a new entity term BEFORE creating nodes that use it",
                "save_to_file": "step1_vocab_entity.json",
                "file_contents": {
                    "term": "YOUR_TERM_HERE",
                    "category": "CONCEPT",
                    "description": "Your description here"
                },
                "valid_categories": ["AGENT","CONCEPT","PROCESS","STATE","PROPERTY","SYSTEM"],
                "curl_command": f"curl -X POST '{BASE}/vocab/entities' -H 'Content-Type: application/json' -d @step1_vocab_entity.json",
                "existing_entity_terms": [e["term"] for e in entities]
            },
            "step_2_add_vocab_predicate": {
                "description": "Add a new predicate BEFORE creating nodes that use it",
                "save_to_file": "step2_vocab_predicate.json",
                "file_contents": {
                    "term": "YOUR_PREDICATE_UPPERCASE",
                    "english": "your predicate in english",
                    "description": "What this predicate means"
                },
                "curl_command": f"curl -X POST '{BASE}/vocab/predicates' -H 'Content-Type: application/json' -d @step2_vocab_predicate.json",
                "existing_predicate_terms": [p["term"] for p in predicates]
            },
            "step_3_add_node": {
                "description": "Add a single node. All three terms must already exist in vocab.",
                "save_to_file": "step3_node.json",
                "file_contents": {
                    "subject": ex_subj,
                    "predicate": ex_pred,
                    "object": ex_obj,
                    "node_type": "EMPIRICAL",
                    "confidence": "SUPPORTED",
                    "domain_tags": ["your_domain"]
                },
                "curl_command": f"curl -X POST '{BASE}/nodes' -H 'Content-Type: application/json' -d @step3_node.json",
                "note": "Response contains the new node UUID — save it for edges/evidence"
            },
            "step_4_add_edge": {
                "description": "Connect two existing nodes with a warranted edge",
                "save_to_file": "step4_edge.json",
                "file_contents": {
                    "source_id": ex_id_1,
                    "target_id": ex_id_2,
                    "relationship": "SUPPORTS",
                    "warrant_text": "Explain why this connection holds",
                    "strength": 0.75
                },
                "valid_relationships": ["SUPPORTS","UNDERMINES","CAUSES","IMPLIES","REQUIRES","CONTRADICTS","CORRELATES_WITH","PREVENTS","DEFINES","EXEMPLIFIES"],
                "curl_command": f"curl -X POST '{BASE}/edges' -H 'Content-Type: application/json' -d @step4_edge.json",
                "real_node_ids": [{"id": n["id"], "statement": n["statement"]} for n in nodes_sample[:8]]
            },
            "step_5_add_evidence": {
                "description": "Attach a citation to an existing node",
                "save_to_file": "step5_evidence.json",
                "file_contents": {
                    "target_id": ex_id_1,
                    "target_type": "NODE",
                    "source_type": "PEER_REVIEWED",
                    "description": "What the source shows, in your own words",
                    "url": "https://doi.org/...",
                    "citation": "Author et al. (2023). Title. Journal.",
                    "year": 2023
                },
                "valid_source_types": ["PEER_REVIEWED","META_ANALYSIS","SYSTEMATIC_REVIEW","OBSERVATIONAL","EXPERT_CONSENSUS","LOGICAL_DERIVATION","ANECDOTAL"],
                "curl_command": f"curl -X POST '{BASE}/evidence' -H 'Content-Type: application/json' -d @step5_evidence.json"
            },
            "step_6_add_challenge": {
                "description": "Challenge a contested node",
                "save_to_file": "step6_challenge.json",
                "file_contents": {
                    "target_id": contested[0]["id"] if contested else ex_id_1,
                    "target_type": "NODE",
                    "ground": "EMPIRICAL",
                    "property_disputed": "confidence",
                    "argument": "Your challenge argument here",
                    "counter_node_id": None
                },
                "valid_grounds": ["EMPIRICAL","LOGICAL","VALUE","SCOPE","DEFINITION"],
                "curl_command": f"curl -X POST '{BASE}/challenges' -H 'Content-Type: application/json' -d @step6_challenge.json",
                "contested_nodes": contested
            },
            "step_7_bulk_propose": {
                "description": "Add nodes, edges, evidence and challenges in one call",
                "save_to_file": "step7_bulk.json",
                "file_contents": {
                    "proposed_by": "your-llm-name",
                    "reasoning": "Describe what you are adding and why",
                    "nodes": [
                        {
                            "subject": ex_subj,
                            "predicate": ex_pred,
                            "object": ex_obj,
                            "node_type": "LOGICAL",
                            "confidence": "SUPPORTED",
                            "domain_tags": ["example"]
                        }
                    ],
                    "edges": [
                        {
                            "source_id": ex_id_1,
                            "target_id": ex_id_2,
                            "relationship": "SUPPORTS",
                            "warrant_text": "Explain the connection",
                            "strength": 0.75
                        }
                    ],
                    "evidence": [],
                    "challenges": []
                },
                "curl_command": f"curl -X POST '{BASE}/llm/propose' -H 'Content-Type: application/json' -d @step7_bulk.json",
                "warnings": [
                    "source_id/target_id in edges[] must be UUID strings, NOT statement text",
                    "If creating nodes and edges in the same call, run nodes first, get their IDs from the response, then run a second call for edges",
                    "All subject/predicate/object terms must exist in vocab before this call"
                ]
            }
        },
        "workflow_summary": (
            "1. GET /llm/curl-guide (this call) to get real IDs. "
            "2. If new vocab needed: save + POST step1/step2 files. "
            "3. Save step3 or step7 JSON, run curl. "
            "4. For edges after new nodes: get UUIDs from step3 response, then run step4. "
            "5. Run steps in order — later steps depend on IDs from earlier responses."
        )
    }
