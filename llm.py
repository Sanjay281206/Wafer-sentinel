"""Root-cause assistant. Two backends behind one function:
  1) local llama.cpp model (GGUF, 1B-3B) constrained to a JSON schema  -> runs on Pi
  2) deterministic rule-based fallback from the knowledge base          -> always works
The LLM never invents numbers: yield stats are computed and passed in."""
import json
from .knowledge import lookup

SCHEMA_KEYS = [
    "summary", "detected_pattern", "likely_causes", "supporting_evidence",
    "manufacturing_processes", "yield_impact", "recommended_action",
    "confidence_note", "limitations",
]

PROMPT = """You are a semiconductor yield engineer. Use ONLY the wafer facts and knowledge-base entries provided.
Wafer facts: {facts}
Knowledge base entries: {kb}
Write a detailed, technically credible JSON report with these keys:
detected_pattern (string), summary (2-3 sentences), likely_causes (ranked list, max 3),
supporting_evidence (list based only on supplied wafer statistics and spatial zones),
manufacturing_processes (list, process names and concise technical meaning),
yield_impact (specific sentence using supplied die and yield numbers),
recommended_action (string with inspection steps and corrective actions),
confidence_note (explicitly distinguish probable hypothesis from confirmed cause),
limitations (list explaining unavailable equipment readings or process logs).
Do not invent measurements, process logs or confirmed root causes.
Output JSON only."""


def _zone_evidence(stats):
    z = stats.get("zone_fail_rate", {})
    center = float(z.get("center", 0))
    mid = float(z.get("mid", 0))
    edge = float(z.get("edge", 0))
    ranked = sorted({"Center": center, "Middle": mid, "Edge": edge}.items(), key=lambda item: item[1], reverse=True)
    evidence = [
        f"The centre zone has a {center:.1%} failure rate.",
        f"The middle zone has a {mid:.1%} failure rate.",
        f"The edge zone has a {edge:.1%} failure rate.",
        f"The highest zone-level concentration is {ranked[0][0]} at {ranked[0][1]:.1%}; "
        "this supports, but does not confirm, the spatial interpretation.",
    ]
    return evidence


def rule_based(patterns, confs, stats, impact, thr=0.6):
    kb = lookup(patterns)
    causes, processes, indications, actions = [], [], [], []
    for pattern in patterns:
        entry = kb.get(pattern, {})
        causes.extend(entry.get("causes", []))
        processes.extend(entry.get("processes", []))
        if entry.get("indication"):
            indications.append(entry["indication"])
        if entry.get("action"):
            actions.append(entry["action"])
    causes = causes[:3]
    processes = list(dict.fromkeys(processes))
    actions = list(dict.fromkeys(actions))
    low = (not confs) or max(confs.values()) < thr if patterns else False
    confidence_note = (
        "Low confidence - manual engineering review recommended."
        if low else "The pattern confidence is acceptable for prioritisation, but it is not confirmation of a process root cause."
    )
    detected = ", ".join(patterns) if patterns else "No systematic pattern"
    total = int(stats["total_die"])
    failed = int(stats["failing_die"])
    yield_pct = float(stats["die_yield"])
    evidence = _zone_evidence(stats)
    evidence.append(
        f"The map contains {total:,} on-wafer dies, with {failed:,} marked defective "
        f"and a calculated map yield of {yield_pct:.1%}."
    )
    if confs:
        evidence.append(
            "Pattern confidences above the configured thresholds: "
            + ", ".join(f"{p} {v:.1%}" for p, v in confs.items())
            + "."
        )
    else:
        evidence.append("No defect label exceeded the configured prediction threshold.")
    summary = (
        f"{detected} is the spatial defect interpretation for this wafer. "
        f"{' '.join(indications[:2])} "
        "This result supports process-specific hypotheses and requires metrology, process-history and tool-health verification."
    )
    return {
        "detected_pattern": detected,
        "summary": summary,
        "likely_causes": causes or ["No systematic pattern was detected; continue baseline monitoring."],
        "supporting_evidence": evidence,
        "manufacturing_processes": processes or ["No pattern-specific process is identified; continue normal process monitoring."],
        "recommended_actions": actions or ["Continue lot monitoring and verify the wafer-map encoding."],
        "recommended_action": " ".join(actions) if actions else "Continue lot monitoring and verify the wafer-map encoding.",
        "yield_impact": (
            f"About {impact['yield_loss_pts']} percentage points below baseline "
            f"(~{impact['lost_die_vs_baseline']} die). {impact['assumption']}."
        ),
        "confidence_note": confidence_note,
        "limitations": [
            "This report uses map geometry and model confidence only.",
            "No equipment readings, process recipes, metrology logs or electrical-test data were supplied.",
            "The identified items are probable hypotheses, not confirmed manufacturing causes.",
        ],
    }


class RootCauseLLM:
    def __init__(self, gguf_path=None, n_ctx=2048, n_threads=4):
        self.llm = None
        if gguf_path:
            from llama_cpp import Llama
            self.llm = Llama(model_path=gguf_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)

    def explain(self, patterns, confs, stats, impact, thr=0.6):
        base = rule_based(patterns, confs, stats, impact, thr)
        if self.llm is None:
            return {**base, "backend": "rule-based"}
        facts = json.dumps({
            "patterns": confs,
            "detected_patterns": patterns,
            "stats": stats,
            "impact": impact,
            "low_confidence_threshold": thr,
        })
        prompt = PROMPT.format(facts=facts, kb=json.dumps(lookup(patterns)), thr=thr)
        try:
            out = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, temperature=0.2, max_tokens=700)
            data = json.loads(out["choices"][0]["message"]["content"])
            assert all(k in data for k in SCHEMA_KEYS)
            data.setdefault("detected_pattern", ", ".join(patterns) if patterns else "No systematic pattern")
            data.setdefault("supporting_evidence", base["supporting_evidence"])
            data.setdefault("manufacturing_processes", base["manufacturing_processes"])
            data.setdefault("recommended_actions", [data.get("recommended_action", base["recommended_action"])])
            data.setdefault("limitations", base["limitations"])
            data["recommended_action"] = data.get("recommended_action", base["recommended_action"])
            return {**data, "backend": "llama.cpp"}
        except Exception:
            return {**base, "backend": "rule-based (llm output invalid)"}
