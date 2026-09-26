"""
Vera AI Bot — magicpin AI Challenge
====================================
A production-grade merchant AI assistant that composes contextual WhatsApp 
messages using the 4-context framework (Category, Merchant, Trigger, Customer).

Powered by Google Gemini for intelligent message composition.
"""

import os
import sys
import time
import json
import re
import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Optional
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment
load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("vera-bot")

# ---------------------------------------------------------------------------
# Gemini Client
# ---------------------------------------------------------------------------

class GeminiClient:
    """Lightweight Google Gemini API client using httpx."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model
        self._url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        log.info(f"Gemini client initialized: model={self.model}")

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.3) -> str:
        """Send a prompt to Gemini and return the text response."""
        import httpx

        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        body = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 2000,
            },
        }
        try:
            resp = httpx.post(self._url, json=body, timeout=28.0)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            log.error(f"Gemini API error: {e}")
            raise


# ---------------------------------------------------------------------------
# Context Store
# ---------------------------------------------------------------------------

class ContextStore:
    """In-memory store for all pushed contexts, keyed by (scope, context_id)."""

    def __init__(self):
        self._data: dict[tuple[str, str], dict] = {}  # (scope, id) -> {version, payload}

    def upsert(self, scope: str, context_id: str, version: int, payload: dict) -> tuple[bool, int | None]:
        """
        Returns (accepted, current_version_if_stale).
        """
        key = (scope, context_id)
        cur = self._data.get(key)
        if cur and cur["version"] >= version:
            return False, cur["version"]
        self._data[key] = {"version": version, "payload": payload}
        return True, None

    def get(self, scope: str, context_id: str) -> dict | None:
        entry = self._data.get((scope, context_id))
        return entry["payload"] if entry else None

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        for (scope, _) in self._data:
            counts[scope] = counts.get(scope, 0) + 1
        return counts

    def get_all_by_scope(self, scope: str) -> dict[str, dict]:
        """Return all payloads for a given scope."""
        return {
            cid: entry["payload"]
            for (s, cid), entry in self._data.items()
            if s == scope
        }


# ---------------------------------------------------------------------------
# Conversation State
# ---------------------------------------------------------------------------

class ConversationStore:
    """Tracks multi-turn conversation history per conversation_id."""

    def __init__(self):
        self._convos: dict[str, list[dict]] = {}
        self._sent_bodies: dict[str, set] = {}  # conv_id -> set of sent bodies (anti-repetition)

    def add_turn(self, conv_id: str, role: str, body: str, meta: dict | None = None):
        self._convos.setdefault(conv_id, []).append({
            "role": role, "body": body, "ts": datetime.now(timezone.utc).isoformat(),
            **(meta or {}),
        })

    def get_history(self, conv_id: str) -> list[dict]:
        return self._convos.get(conv_id, [])

    def mark_sent(self, conv_id: str, body: str):
        self._sent_bodies.setdefault(conv_id, set()).add(body.strip().lower())

    def was_sent_before(self, conv_id: str, body: str) -> bool:
        return body.strip().lower() in self._sent_bodies.get(conv_id, set())

    def get_active_conv_ids(self) -> set[str]:
        return set(self._convos.keys())


# ---------------------------------------------------------------------------
# Composer Prompt Engine
# ---------------------------------------------------------------------------

COMPOSER_SYSTEM = """You are Vera, magicpin's merchant AI assistant that talks to merchants and their customers over WhatsApp.

YOUR CORE IDENTITY:
- You are a knowledgeable peer/colleague, NOT a salesperson
- You use Hindi-English code-mix naturally when the merchant's language includes "hi"
- You are concise, specific, and action-oriented
- You NEVER fabricate data — only use what's provided in the context
- You NEVER use generic promotional language ("AMAZING DEAL!", "Flat 30% off")

VOICE RULES BY CATEGORY:
- Dentists: Clinical-peer tone, technical vocabulary OK, source citations, use "Dr." prefix, taboos: "cure", "guaranteed", "100% safe"
- Salons: Warm, friendly, practical, trend-aware, use owner's first name
- Restaurants: Operator-to-operator, numbers-driven, local-event aware
- Gyms: Coaching, motivational, data-backed, use first name
- Pharmacies: Trustworthy, precise, compliance-aware, use owner's first name

MESSAGE COMPOSITION RULES:
1. Lead with the SPECIFIC hook — a number, date, research citation, or verifiable fact
2. Connect to WHY NOW — the trigger event that prompted this message
3. Personalize to THIS merchant — use their actual data, not generic statements
4. End with a SINGLE clear CTA — binary (YES/STOP) for action triggers, open-ended for information
5. Keep it SHORT — WhatsApp messages. No preambles. No "I hope you're doing well"
6. Use service+price format ("Dental Cleaning @ ₹299") NOT discount format ("30% off")
7. Match the merchant's language preferences
8. NEVER re-introduce yourself after the first message in a conversation

COMPULSION LEVERS TO USE (pick 1-2 per message):
- Specificity/verifiability — concrete number, date, headline
- Loss aversion — "you're missing X"
- Social proof — "3 dentists in your area did Y"
- Effort externalization — "I've drafted X — just say go"
- Curiosity — "want to see who?"
- Reciprocity — "I noticed Y, thought you'd want to know"
- Single binary commitment — Reply YES / STOP

ANTI-PATTERNS TO AVOID:
- Multiple CTAs in one message
- Buried call-to-action (CTA must be in the last sentence)
- Long preambles ("I hope you're doing well. I'm reaching out today to…")
- Generic offers when service+price is available
- Hallucinated data not in the provided context
"""

def build_compose_prompt(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None,
    conversation_history: list[dict] | None = None,
) -> str:
    """Build the composition prompt from the 4 contexts."""

    # Extract key merchant info
    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    sub = merchant.get("subscription", {})
    signals = merchant.get("signals", [])
    offers = merchant.get("offers", [])
    active_offers = [o for o in offers if o.get("status") == "active"]
    conv_hist = merchant.get("conversation_history", [])
    cust_agg = merchant.get("customer_aggregate", {})
    review_themes = merchant.get("review_themes", [])

    # Category info
    voice = category.get("voice", {})
    peer_stats = category.get("peer_stats", {})
    digest = category.get("digest", [])
    offer_catalog = category.get("offer_catalog", [])
    seasonal_beats = category.get("seasonal_beats", [])
    trend_signals = category.get("trend_signals", [])
    patient_content = category.get("patient_content_library", [])

    # Find referenced digest items from trigger
    trigger_payload = trigger.get("payload", {})
    top_item_id = trigger_payload.get("top_item_id") or trigger_payload.get("digest_item_id")
    referenced_digest = None
    if top_item_id:
        for d in digest:
            if d.get("id") == top_item_id:
                referenced_digest = d
                break

    # Determine scope
    is_customer_facing = trigger.get("scope") == "customer" and customer is not None

    prompt = f"""=== COMPOSE A WHATSAPP MESSAGE ===

SCOPE: {"CUSTOMER-FACING (sent from merchant's number, on their behalf)" if is_customer_facing else "MERCHANT-FACING (Vera to merchant)"}

--- CATEGORY CONTEXT ---
Category: {category.get("slug", "unknown")} ({category.get("display_name", "")})
Voice/Tone: {voice.get("tone", "professional")}
Vocabulary allowed: {", ".join(voice.get("vocab_allowed", [])[:8])}
Vocabulary TABOO (never use): {", ".join(voice.get("vocab_taboo", [])[:5])}
Peer benchmarks: avg_rating={peer_stats.get("avg_rating", "?")}, avg_reviews={peer_stats.get("avg_review_count", "?")}, avg_CTR={peer_stats.get("avg_ctr", "?")}, avg_views_30d={peer_stats.get("avg_views_30d", "?")}
Category offer catalog: {json.dumps([o.get("title") for o in offer_catalog[:5]])}
Seasonal beats: {json.dumps(seasonal_beats[:3])}
Trend signals: {json.dumps(trend_signals[:3])}
"""

    if referenced_digest:
        prompt += f"""
Referenced digest item:
  Title: {referenced_digest.get("title", "")}
  Source: {referenced_digest.get("source", "")}
  Summary: {referenced_digest.get("summary", "")}
  Actionable: {referenced_digest.get("actionable", "")}
  Trial N: {referenced_digest.get("trial_n", "N/A")}
  Patient segment: {referenced_digest.get("patient_segment", "N/A")}
"""

    if patient_content and is_customer_facing:
        prompt += f"\nPatient content library: {json.dumps([p.get('title') for p in patient_content[:3]])}\n"

    prompt += f"""
--- MERCHANT CONTEXT ---
Name: {identity.get("name", "Unknown")}
Owner: {identity.get("owner_first_name", "Unknown")}
City: {identity.get("city", "?")}, Locality: {identity.get("locality", "?")}
Languages: {identity.get("languages", ["en"])}
Verified GBP: {identity.get("verified", False)}
Established: {identity.get("established_year", "?")}
Subscription: {sub.get("status", "?")} ({sub.get("plan", "?")}) — {sub.get("days_remaining", "?")} days remaining
Performance (30d): views={perf.get("views", "?")}, calls={perf.get("calls", "?")}, directions={perf.get("directions", "?")}, CTR={perf.get("ctr", "?")}
7-day delta: views {perf.get("delta_7d", {}).get("views_pct", "?")}%, calls {perf.get("delta_7d", {}).get("calls_pct", "?")}%
Active offers: {json.dumps([o.get("title") for o in active_offers]) if active_offers else "NONE"}
Customer aggregate: {json.dumps(cust_agg)}
Signals: {json.dumps(signals)}
Review themes: {json.dumps(review_themes[:3]) if review_themes else "None"}
"""

    # Add conversation history with Vera
    if conv_hist:
        prompt += "\nRecent conversation with Vera:\n"
        for turn in conv_hist[-4:]:
            prompt += f"  [{turn.get('from', '?')}] {turn.get('body', '')[:150]}\n"
            prompt += f"    Engagement: {turn.get('engagement', '?')}\n"

    prompt += f"""
--- TRIGGER CONTEXT ---
Kind: {trigger.get("kind", "unknown")}
Source: {trigger.get("source", "?")} / Scope: {trigger.get("scope", "?")}
Urgency: {trigger.get("urgency", "?")} (1=low, 5=critical)
Payload: {json.dumps(trigger_payload)}
Suppression key: {trigger.get("suppression_key", "")}
"""

    if is_customer_facing and customer:
        cust_identity = customer.get("identity", {})
        cust_rel = customer.get("relationship", {})
        prompt += f"""
--- CUSTOMER CONTEXT ---
Name: {cust_identity.get("name", "Customer")}
Language pref: {cust_identity.get("language_pref", "en")}
State: {customer.get("state", "?")}
Relationship: first_visit={cust_rel.get("first_visit", "?")}, last_visit={cust_rel.get("last_visit", "?")}, total_visits={cust_rel.get("visits_total", "?")}
Services received: {json.dumps(cust_rel.get("services_received", []))}
Preferences: {json.dumps(customer.get("preferences", {}))}
Consent scope: {json.dumps(customer.get("consent", {}).get("scope", []))}
"""

    # Add any active conversation for this thread
    if conversation_history:
        prompt += "\n--- CONVERSATION SO FAR ---\n"
        for turn in conversation_history[-6:]:
            prompt += f"  [{turn.get('role', '?')}]: {turn.get('body', '')[:200]}\n"

    prompt += """
--- YOUR OUTPUT ---
Respond with ONLY a JSON object (no markdown fences, no extra text):
{
  "body": "<the WhatsApp message body>",
  "cta": "<'binary_yes_stop' | 'open_ended' | 'none'>",
  "send_as": "<'vera' for merchant-facing | 'merchant_on_behalf' for customer-facing>",
  "suppression_key": "<copy from trigger or create based on context>",
  "rationale": "<1-2 sentences: why this message, what trigger drove it, what compulsion lever>"
}
"""

    return prompt


def build_reply_prompt(
    merchant: dict,
    category: dict,
    conversation_history: list[dict],
    merchant_message: str,
) -> str:
    """Build a prompt for replying to a merchant's message in an ongoing conversation."""

    identity = merchant.get("identity", {})

    # Auto-reply detection
    auto_reply_indicators = [
        "thank you for contacting",
        "our team will respond",
        "automated",
        "auto-reply",
        "we will get back",
        "thanks for reaching out",
        "currently unavailable",
        "away message",
    ]
    is_likely_auto = any(ind in merchant_message.lower() for ind in auto_reply_indicators)

    # Count repeated auto-replies
    auto_count = 0
    if is_likely_auto:
        for turn in conversation_history:
            if turn.get("role") == "merchant":
                body_lower = turn.get("body", "").lower().strip()
                if any(ind in body_lower for ind in auto_reply_indicators):
                    auto_count += 1

    # Intent detection
    intent_phrases = [
        "ok let", "lets do", "go ahead", "yes", "proceed", "I want to join",
        "sign me up", "do it", "chalein", "haan", "kar do", "shuru karo",
        "start", "ready", "confirm",
    ]
    has_action_intent = any(p in merchant_message.lower() for p in intent_phrases)

    # Hostility detection
    hostile_phrases = [
        "stop", "spam", "useless", "don't message", "block", "annoying",
        "waste of time", "not interested", "unsubscribe", "band karo",
        "mat bhejo", "bakwas",
    ]
    is_hostile = any(p in merchant_message.lower() for p in hostile_phrases)

    prompt = f"""=== REPLY TO MERCHANT MESSAGE ===

Merchant: {identity.get("name", "Unknown")} ({identity.get("owner_first_name", "")})
Category: {merchant.get("category_slug", "unknown")}
Languages: {identity.get("languages", ["en"])}

CONVERSATION SO FAR:
"""
    for turn in conversation_history[-6:]:
        role_label = "VERA" if turn.get("role") == "vera" else "MERCHANT"
        prompt += f"[{role_label}]: {turn.get('body', '')[:200]}\n"

    prompt += f"""
LATEST MERCHANT MESSAGE: "{merchant_message}"

DETECTION FLAGS:
- Likely auto-reply: {is_likely_auto} (count so far: {auto_count})
- Action intent detected: {has_action_intent}
- Hostile/unsubscribe: {is_hostile}

RULES FOR YOUR RESPONSE:
1. If this is an AUTO-REPLY (canned WA Business message) seen 2+ times: respond with action="end", apologize briefly and exit gracefully
2. If auto-reply detected for FIRST time: try one more message to reach the real person, keep it brief
3. If merchant shows ACTION INTENT ("yes", "let's do it", "go ahead"): switch IMMEDIATELY to action mode. DO NOT ask another qualifying question.
4. If merchant is HOSTILE or says "stop"/"not interested": respond with action="end", be polite and respectful
5. If merchant asks a QUESTION: answer it concisely with data from context
6. If merchant engages normally: continue the conversation naturally, advance toward next value step

Respond with ONLY a JSON object (no markdown fences):
{{
  "action": "<'send' | 'wait' | 'end'>",
  "body": "<your reply (only if action=send)>",
  "cta": "<'binary_yes_stop' | 'open_ended' | 'none'>",
  "rationale": "<why this action, 1 sentence>",
  "wait_seconds": <only if action=wait, number of seconds>
}}
"""
    return prompt


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(title="Vera Bot — magicpin AI Challenge", version="1.0.0")
START_TIME = time.time()

# Stores
store = ContextStore()
conversations = ConversationStore()

# Gemini client (lazy init)
_gemini: GeminiClient | None = None

def get_gemini() -> GeminiClient:
    global _gemini
    if _gemini is None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")
        _gemini = GeminiClient(api_key, model)
    return _gemini


def parse_llm_json(text: str) -> dict:
    """Robustly parse JSON from LLM output, handling markdown fences."""
    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text.strip())
    # Find JSON object
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Fallback: try the whole thing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.error(f"Failed to parse LLM JSON: {text[:200]}")
        return {}


def compose_message(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    Compose a message using the 4-context framework + Gemini.
    Falls back to template-based composition if Gemini fails.
    """
    try:
        gemini = get_gemini()
        prompt = build_compose_prompt(category, merchant, trigger, customer, conversation_history)
        raw = gemini.complete(prompt, COMPOSER_SYSTEM, temperature=0.3)
        result = parse_llm_json(raw)
        if result and result.get("body"):
            return result
    except Exception as e:
        log.warning(f"LLM composition failed, using fallback: {e}")

    # ── Fallback: deterministic template composition ──
    return compose_fallback(category, merchant, trigger, customer)


def compose_fallback(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None,
) -> dict:
    """Deterministic template-based fallback composer."""
    identity = merchant.get("identity", {})
    name = identity.get("name", "there")
    owner = identity.get("owner_first_name", "")
    perf = merchant.get("performance", {})
    cat_slug = category.get("slug", "business")
    trigger_kind = trigger.get("kind", "")
    trigger_payload = trigger.get("payload", {})
    is_customer = trigger.get("scope") == "customer" and customer is not None

    salutation = f"Dr. {owner}" if cat_slug == "dentists" and owner else owner or name

    if is_customer and customer:
        cust_name = customer.get("identity", {}).get("name", "there")
        if trigger_kind == "recall_due":
            service = trigger_payload.get("service_due", "checkup")
            slots = trigger_payload.get("available_slots", [])
            slot_text = " ya ".join([s.get("label", "") for s in slots[:2]]) if slots else "this week"
            body = (
                f"Hi {cust_name}, {name} here 🙏 "
                f"Aapka {service.replace('_', ' ')} due hai. "
                f"Slots available: {slot_text}. "
                f"Reply with your preferred time or tell us what works!"
            )
            return {"body": body, "cta": "open_ended", "send_as": "merchant_on_behalf",
                    "suppression_key": trigger.get("suppression_key", ""), "rationale": f"Recall reminder for {cust_name}"}
        elif trigger_kind == "chronic_refill_due":
            mols = trigger_payload.get("molecule_list", [])
            mol_text = ", ".join(mols[:3]) if mols else "your regular medicines"
            body = (
                f"Hi {cust_name}, {name} se. "
                f"Aapki {mol_text} ki refill due hone wali hai. "
                f"Kya hum delivery schedule kar dein? Reply YES for home delivery."
            )
            return {"body": body, "cta": "binary_yes_stop", "send_as": "merchant_on_behalf",
                    "suppression_key": trigger.get("suppression_key", ""), "rationale": f"Chronic refill reminder"}
        else:
            body = (
                f"Hi {cust_name}, {name} ki taraf se. "
                f"Aapke liye ek update hai — please visit us soon! "
                f"Reply for details."
            )
            return {"body": body, "cta": "open_ended", "send_as": "merchant_on_behalf",
                    "suppression_key": trigger.get("suppression_key", ""), "rationale": "General customer outreach"}

    # Merchant-facing fallback
    views = perf.get("views", 0)
    calls = perf.get("calls", 0)
    ctr = perf.get("ctr", 0)

    if trigger_kind == "perf_dip":
        metric = trigger_payload.get("metric", "calls")
        delta = trigger_payload.get("delta_pct", -0.2)
        body = (
            f"{salutation}, aapke {metric} mein {abs(int(delta * 100))}% drop dikha last 7 days mein. "
            f"30d performance: {views} views, {calls} calls, CTR {ctr}. "
            f"Want me to check kya ho raha hai aur suggest kuch quick fixes?"
        )
        return {"body": body, "cta": "open_ended", "send_as": "vera",
                "suppression_key": trigger.get("suppression_key", ""), "rationale": "Perf dip alert with data"}

    elif trigger_kind == "perf_spike":
        metric = trigger_payload.get("metric", "views")
        delta = trigger_payload.get("delta_pct", 0.15)
        body = (
            f"{salutation}, good news 📈 Aapke {metric} {int(delta * 100)}% up hain last 7 days! "
            f"Total 30d: {views} views, {calls} calls. "
            f"Want me to analyze what's driving this so we can keep the momentum?"
        )
        return {"body": body, "cta": "open_ended", "send_as": "vera",
                "suppression_key": trigger.get("suppression_key", ""), "rationale": "Perf spike celebration + next step"}

    elif trigger_kind == "research_digest":
        digest = category.get("digest", [])
        top_item = None
        top_item_id = trigger_payload.get("top_item_id")
        for d in digest:
            if d.get("id") == top_item_id:
                top_item = d
                break
        if top_item:
            body = (
                f"{salutation}, {top_item.get('source', 'new research')} mein ek relevant finding — "
                f"{top_item.get('title', '')}. "
                f"{top_item.get('summary', '')[:120]}. "
                f"Want me to pull the details?"
            )
        else:
            body = f"{salutation}, this week's {cat_slug} digest has some relevant findings. Want me to share the highlights?"
        return {"body": body, "cta": "open_ended", "send_as": "vera",
                "suppression_key": trigger.get("suppression_key", ""), "rationale": "Research digest push with source citation"}

    elif trigger_kind in ("renewal_due", "winback_eligible"):
        days = trigger_payload.get("days_remaining", trigger_payload.get("days_since_expiry", "?"))
        body = (
            f"{salutation}, aapki subscription ka status check kiya — "
            f"{'expires in ' + str(days) + ' days' if trigger_kind == 'renewal_due' else 'expired ' + str(days) + ' days ago'}. "
            f"Profile maintenance {'continues' if trigger_kind == 'renewal_due' else 'is paused'}. "
            f"Want to discuss renewal options?"
        )
        return {"body": body, "cta": "open_ended", "send_as": "vera",
                "suppression_key": trigger.get("suppression_key", ""), "rationale": "Subscription status alert"}

    elif trigger_kind == "milestone_reached":
        metric = trigger_payload.get("metric", "reviews")
        value = trigger_payload.get("value_now", "?")
        milestone = trigger_payload.get("milestone_value", "?")
        body = (
            f"{salutation}, you're at {value} {metric} — {milestone} milestone bahut close hai! 🎉 "
            f"Want me to draft a post celebrating when you hit {milestone}?"
        )
        return {"body": body, "cta": "open_ended", "send_as": "vera",
                "suppression_key": trigger.get("suppression_key", ""), "rationale": "Milestone proximity motivation"}

    elif trigger_kind == "festival_upcoming":
        festival = trigger_payload.get("festival", "upcoming festival")
        days_until = trigger_payload.get("days_until", "?")
        body = (
            f"{salutation}, {festival} {days_until} days away. "
            f"Want me to draft a festive GBP post + offer for your profile?"
        )
        return {"body": body, "cta": "open_ended", "send_as": "vera",
                "suppression_key": trigger.get("suppression_key", ""), "rationale": "Festival prep nudge"}

    elif trigger_kind == "competitor_opened":
        comp = trigger_payload.get("competitor_name", "a new competitor")
        dist = trigger_payload.get("distance_km", "nearby")
        their_offer = trigger_payload.get("their_offer", "")
        body = (
            f"{salutation}, FYI — {comp} opened {dist}km away"
            f"{' with ' + their_offer if their_offer else ''}. "
            f"Aapki profile strong hai — want me to compare and suggest any tweaks?"
        )
        return {"body": body, "cta": "open_ended", "send_as": "vera",
                "suppression_key": trigger.get("suppression_key", ""), "rationale": "Competitor alert with comparison offer"}

    elif trigger_kind == "dormant_with_vera":
        days = trigger_payload.get("days_since_last_merchant_message", "?")
        body = (
            f"{salutation}, it's been {days} days! "
            f"Aapki profile ka quick health-check kiya — {views} views last 30d. "
            f"Want a summary of what's changed?"
        )
        return {"body": body, "cta": "open_ended", "send_as": "vera",
                "suppression_key": trigger.get("suppression_key", ""), "rationale": "Re-engagement after dormancy"}

    else:
        # Generic but still data-anchored
        body = (
            f"{salutation}, quick update — aapki 30d stats: "
            f"{views} views, {calls} calls, CTR {ctr}. "
            f"Kuch specific improve karna chahein toh batayein!"
        )
        return {"body": body, "cta": "open_ended", "send_as": "vera",
                "suppression_key": trigger.get("suppression_key", ""), "rationale": f"Fallback: {trigger_kind} trigger handling"}


def compose_reply(
    merchant: dict,
    category: dict,
    merchant_message: str,
    conversation_history: list[dict],
) -> dict:
    """Compose a reply to a merchant's message."""

    # ── Fast-path heuristic detections ──
    auto_reply_indicators = [
        "thank you for contacting", "our team will respond", "automated",
        "auto-reply", "we will get back", "thanks for reaching out",
        "currently unavailable", "away message",
    ]
    is_auto = any(ind in merchant_message.lower() for ind in auto_reply_indicators)
    auto_count = sum(
        1 for t in conversation_history if t.get("role") == "merchant"
        and any(ind in t.get("body", "").lower() for ind in auto_reply_indicators)
    )

    if is_auto and auto_count >= 2:
        identity = merchant.get("identity", {})
        return {
            "action": "end",
            "body": f"Samajh gayi — auto-reply chal raha hai. Main {identity.get('owner_first_name', 'owner')} se directly connect kar lungi. Best wishes! 🙂",
            "cta": "none",
            "rationale": f"Auto-reply detected {auto_count+1} times — gracefully exiting",
        }

    hostile_phrases = [
        "stop", "spam", "useless", "don't message", "block", "not interested",
        "unsubscribe", "band karo", "mat bhejo",
    ]
    is_hostile = any(p in merchant_message.lower() for p in hostile_phrases)

    if is_hostile:
        return {
            "action": "end",
            "body": "Bilkul, noted. Aage se message nahi bhejungi. Agar kabhi zaroorat ho toh reach out kar sakte hain. Best wishes! 🙏",
            "cta": "none",
            "rationale": "Merchant expressed disinterest/hostility — exiting gracefully",
        }

    # ── LLM-based reply ──
    try:
        gemini = get_gemini()
        prompt = build_reply_prompt(merchant, category, conversation_history, merchant_message)
        raw = gemini.complete(prompt, COMPOSER_SYSTEM, temperature=0.3)
        result = parse_llm_json(raw)
        if result and result.get("action"):
            return result
    except Exception as e:
        log.warning(f"LLM reply failed, using fallback: {e}")

    # Fallback reply
    intent_phrases = ["ok let", "lets do", "go ahead", "yes", "proceed", "haan", "kar do"]
    if any(p in merchant_message.lower() for p in intent_phrases):
        return {
            "action": "send",
            "body": "Done — proceeding now. I'll share the details in a minute. 👍",
            "cta": "none",
            "rationale": "Merchant expressed intent — switching to action mode",
        }

    return {
        "action": "send",
        "body": "Got it, noted. Working on it — will share an update shortly.",
        "cta": "open_ended",
        "rationale": "Generic acknowledgment + advancement",
    }


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

class ContextPush(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickRequest(BaseModel):
    now: str
    available_triggers: list[str] = Field(default_factory=list)


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": store.counts(),
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Saksh",
        "team_members": ["Saksh"],
        "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        "approach": "4-context LLM composer with trigger-routed prompt variants, auto-reply detection, intent transitions, and deterministic fallback templates",
        "contact_email": "saksh@example.com",
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/context")
async def push_context(body: ContextPush):
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope", "details": f"Must be one of {valid_scopes}"},
        )

    accepted, stale_version = store.upsert(body.scope, body.context_id, body.version, body.payload)

    if not accepted:
        return JSONResponse(
            status_code=409,
            content={"accepted": False, "reason": "stale_version", "current_version": stale_version},
        )

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat() + "Z",
    }


@app.post("/v1/tick")
async def tick(body: TickRequest):
    actions = []
    processed_merchants: set[str] = set()  # One action per merchant per tick

    for trg_id in body.available_triggers:
        trigger = store.get("trigger", trg_id)
        if not trigger:
            continue

        merchant_id = trigger.get("merchant_id")
        if not merchant_id or merchant_id in processed_merchants:
            continue

        merchant = store.get("merchant", merchant_id)
        if not merchant:
            continue

        cat_slug = merchant.get("category_slug") or trigger.get("payload", {}).get("category")
        category = store.get("category", cat_slug) if cat_slug else None
        if not category:
            continue

        # Get customer if this is customer-scoped
        customer_id = trigger.get("customer_id")
        customer = store.get("customer", customer_id) if customer_id else None

        # Generate a unique conversation ID
        conv_id = f"conv_{merchant_id}_{trg_id}_{int(time.time())}"

        # Check for existing conversations to avoid spamming
        try:
            composed = compose_message(category, merchant, trigger, customer)
        except Exception as e:
            log.error(f"Composition error for {trg_id}: {e}\n{traceback.format_exc()}")
            continue

        if not composed or not composed.get("body"):
            continue

        body_text = composed["body"]

        # Anti-repetition: skip if we've already sent this exact body
        if conversations.was_sent_before(conv_id, body_text):
            log.info(f"Skipping duplicate body for {conv_id}")
            continue

        # Record the conversation
        conversations.add_turn(conv_id, "vera", body_text, {"trigger_id": trg_id})
        conversations.mark_sent(conv_id, body_text)
        processed_merchants.add(merchant_id)

        is_customer_facing = trigger.get("scope") == "customer" and customer is not None

        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed.get("send_as", "merchant_on_behalf" if is_customer_facing else "vera"),
            "trigger_id": trg_id,
            "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
            "template_params": [
                merchant.get("identity", {}).get("name", ""),
                trigger.get("kind", ""),
                composed.get("body", "")[:50],
            ],
            "body": body_text,
            "cta": composed.get("cta", "open_ended"),
            "suppression_key": composed.get("suppression_key", trigger.get("suppression_key", "")),
            "rationale": composed.get("rationale", "Composed from 4-context framework"),
        })

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyRequest):
    # Record the merchant's message
    conversations.add_turn(body.conversation_id, body.from_role, body.message)

    # Get relevant contexts
    merchant = store.get("merchant", body.merchant_id) if body.merchant_id else {}
    merchant = merchant or {}
    cat_slug = merchant.get("category_slug", "")
    category = store.get("category", cat_slug) if cat_slug else {}
    category = category or {}

    # Get conversation history
    history = conversations.get_history(body.conversation_id)

    # Compose the reply
    result = compose_reply(merchant, category, body.message, history)

    # Record our reply if sending
    if result.get("action") == "send" and result.get("body"):
        conversations.add_turn(body.conversation_id, "vera", result["body"])
        conversations.mark_sent(body.conversation_id, result["body"])

    # Ensure valid response
    action = result.get("action", "send")
    if action not in ("send", "wait", "end"):
        action = "send"

    response = {"action": action, "rationale": result.get("rationale", "")}

    if action == "send":
        response["body"] = result.get("body", "Noted, working on it.")
        response["cta"] = result.get("cta", "open_ended")
    elif action == "wait":
        response["wait_seconds"] = result.get("wait_seconds", 300)
        response["rationale"] = result.get("rationale", "Backing off briefly")
    elif action == "end":
        response["body"] = result.get("body", "")
        response["rationale"] = result.get("rationale", "Conversation ended")

    return response


# ---------------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    log.error(f"Unhandled error: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    log.info(f"Starting Vera bot on port {port}")
    uvicorn.run("bot:app", host="0.0.0.0", port=port, reload=False)
