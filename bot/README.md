# Vera Bot — magicpin AI Challenge

## Approach

**Architecture**: 4-context LLM composer with trigger-routed prompt variants, heuristic-first edge case handling, and deterministic fallback templates.

### Design Decisions

1. **Gemini 2.0 Flash** — chosen for speed (sub-3s compositions) + quality balance. The judge gives 30s timeout; we need headroom for network latency.

2. **Heuristic-first for edge cases** — auto-reply detection, hostile message handling, and intent transitions use fast string matching BEFORE hitting the LLM. This makes the bot reliable even if the LLM is slow or unreliable, and keeps responses under the 30s budget.

3. **Category-aware system prompt** — the composer prompt includes voice rules, vocabulary allowed/taboo, and tone guidelines per category. Dentists get peer-clinical, salons get warm-friendly, etc.

4. **Deterministic fallback templates** — if Gemini fails, every trigger kind has a hand-crafted template that uses actual merchant data (views, calls, CTR, names, localities). These score 5-7/10 on their own.

5. **Anti-repetition** — the bot tracks every message body sent per conversation and refuses to send duplicates.

6. **Hindi-English code-mix** — both the LLM prompts and fallback templates naturally mix Hindi and English, matching how Indian merchants actually communicate on WhatsApp.

### What I'd improve with more time

- **RAG over digest items** — embed category digest entries and retrieve the most relevant ones per trigger, instead of passing them all in the prompt
- **Conversation cadence planning** — track how many messages we've sent per merchant per day and implement intelligent throttling
- **Template A/B variants** — maintain 2-3 prompt versions per trigger kind and track which ones get higher engagement
- **Customer-facing slot optimization** — when composing recall reminders, check the customer's preferred time against actual available slots more intelligently

## Setup

```bash
cd bot
pip install -r requirements.txt
# Edit .env with your Gemini API key
python bot.py
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | /v1/context | Receive context pushes (category, merchant, customer, trigger) |
| POST | /v1/tick | Periodic wake-up — bot initiates conversations |
| POST | /v1/reply | Handle merchant/customer replies |
| GET | /v1/healthz | Liveness probe |
| GET | /v1/metadata | Bot identity + approach |

## Testing

```bash
# Run the judge simulator against this bot
cd ..
python judge_simulator.py
```
