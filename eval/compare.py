"""Print comparison table from saved eval JSONL results."""
import json
from pathlib import Path

results_dir = Path("eval/results")
summary = {}

for model_dir in sorted(results_dir.iterdir()):
    if not model_dir.is_dir():
        continue
    files = sorted(model_dir.glob("eval_*.jsonl"))
    if not files:
        continue
    latest = files[-1]
    rows = [json.loads(l) for l in latest.read_text(encoding="utf-8").splitlines() if l.strip()]
    kw  = [r["keyword_recall"] for r in rows]
    f1 = [r["word_f1"] for r in rows]
    lat = [r["latency_s"] for r in rows]
    summary[model_dir.name] = {
        "kw": sum(kw) / len(kw),
        "f1": sum(f1) / len(f1),
        "lat": sum(lat) / len(lat),
        "n": len(rows),
    }

print()
print("=" * 70)
print(f"  {'Model':<42} {'KW Recall':>10} {'Word F1':>9} {'Latency':>9}")
print("-" * 70)
for m, s in summary.items():
    print(f"  {m:<42} {s['kw']:>10.4f} {s['f1']:>9.4f} {s['lat']:>8.2f}s")
print("=" * 70)

best_kw  = max(summary, key=lambda m: summary[m]["kw"])
best_f1  = max(summary, key=lambda m: summary[m]["f1"])
best_lat = min(summary, key=lambda m: summary[m]["lat"])
print(f"  Best KW Recall : {best_kw}  ({summary[best_kw]['kw']:.4f})")
print(f"  Best Word F1   : {best_f1}  ({summary[best_f1]['f1']:.4f})")
print(f"  Fastest        : {best_lat}  ({summary[best_lat]['lat']:.2f}s)")
print("=" * 70)
