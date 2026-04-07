from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

INPUT_JSONL = Path("eval/results/stage2/gen_prompt_ablation_v3/eval_20260313_115651.jsonl")
OUT_JSON = Path("eval/results/stage2/genbad9_case_analysis.json")
OUT_PROMPT = Path("eval/results/stage2/genbad9_prompt_patch_v1.txt")


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _extract_missing_signals(judge_reason: str) -> list[str]:
    reason = judge_reason.lower()
    signals: list[str] = []
    if "omet" in reason or "omise" in reason or "omises" in reason:
        signals.append("omission")
    if "se concentre" in reason or "limite" in reason:
        signals.append("scope_too_narrow")
    if "n\u2019" in judge_reason or "n'" in judge_reason:
        if "ne mentionne" in reason:
            signals.append("missing_mentions")
    if "g\u00e9n\u00e9rale" in reason or "g\u00e9n\u00e9ral" in reason:
        signals.append("missed_general_rule")
    if "secteur" in reason and "priv" in reason:
        signals.append("public_private_confusion")
    if "indemnit" in reason:
        signals.append("missing_indemnities")
    if "condition" in reason:
        signals.append("missing_conditions")
    if "d\u00e9lai" in reason or "recours" in reason:
        signals.append("missing_timeline_or_remedies")
    return signals


def _topic_from_question(question: str) -> str:
    question_l = question.lower()
    if "retraite" in question_l:
        return "retraite"
    if "formation" in question_l:
        return "formation"
    if "cotisations" in question_l:
        return "cotisations_sociales"
    if "discrimination" in question_l:
        return "discrimination"
    if "temps partiel" in question_l:
        return "temps_partiel"
    if "repr\u00e9sentation du personnel" in question_l:
        return "representants_personnel"
    if "prime" in question_l or "r\u00e9mun\u00e9ration" in question_l:
        return "primes_remuneration"
    if "fonctionnaires" in question_l:
        return "fonction_publique"
    return "other"


def build_prompt_patch(top_signals: list[str]) -> str:
    checklist = [
        "- Couvre d'abord le cadre GENERAL applicable au salarie en entreprise privee, puis les regimes speciaux seulement si la question les vise explicitement.",
        "- Avant de finaliser, verifie cette checklist: (1) regle generale, (2) conditions d'application, (3) effets concrets (droits/obligations/indemnites), (4) limites/exceptions, (5) delais/recours si presents.",
        "- Si les sources ne couvrent qu'un sous-regime (ex: secteur public, profession reglementee), indique-le explicitement et n'en fais pas la regle generale.",
        "- N'omets pas les points structurels demandes dans la question (ex: cotisations maladie/AT/retraite/allocation; formation continue vs apprentissage; nomination ET avancement).",
        "- Si une information attendue n'est pas dans les sources, ajoute: \"Element non precise par les sources fournies.\"",
    ]
    if "missing_timeline_or_remedies" in top_signals:
        checklist.append("- Quand present dans les sources, ajoute une ligne concise sur les delais, voies de recours ou sanctions.")

    return (
        "PATCH_PROMPT_V1\n"
        "Ajoute les consignes suivantes a la fin du SYSTEM_PROMPT_V3, sans changer les regles de citation:\n"
        + "\n".join(checklist)
    )


def main() -> None:
    rows = _load_rows(INPUT_JSONL)
    incorrect = [r for r in rows if r.get("judge_label") == "incorrect"]

    signal_counter: Counter[str] = Counter()
    topic_counter: Counter[str] = Counter()
    case_summaries: list[dict] = []

    for row in incorrect:
        reason = row.get("judge_reason", "")
        signals = _extract_missing_signals(reason)
        signal_counter.update(signals)
        topic = _topic_from_question(row.get("question", ""))
        topic_counter.update([topic])
        case_summaries.append(
            {
                "question_id": row.get("question_id"),
                "topic": topic,
                "judge_score": row.get("judge_score"),
                "keyword_recall": row.get("keyword_recall"),
                "context_coverage": row.get("context_coverage"),
                "signals": signals,
                "judge_reason": reason,
            }
        )

    top_signals = [signal for signal, _ in signal_counter.most_common(6)]
    prompt_patch = build_prompt_patch(top_signals)

    report = {
        "input_file": str(INPUT_JSONL),
        "total_cases": len(rows),
        "incorrect_cases": len(incorrect),
        "signal_counts": dict(signal_counter),
        "topic_counts": dict(topic_counter),
        "top_signals": top_signals,
        "cases": case_summaries,
        "recommended_prompt_patch_file": str(OUT_PROMPT),
    }

    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_PROMPT.write_text(prompt_patch + "\n", encoding="utf-8")
    print(f"Wrote analysis: {OUT_JSON}")
    print(f"Wrote prompt patch: {OUT_PROMPT}")


if __name__ == "__main__":
    main()
