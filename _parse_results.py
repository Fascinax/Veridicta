import csv

rows = []
with open('autoeval/results.tsv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        rows.append(row)

print(f'Total experiments: {len(rows)}')
print()
print(f"{'exp':>3} {'retriever':<18} {'k':>2} {'KW':>6} {'F1':>6} {'CitFaith':>8} {'CtxCov':>7} {'Lat':>6} {'score':>7}")
print('-'*75)
for r in rows:
    print(f"{r['exp_id']:>3} {r['retriever']:<18} {r['k']:>2} {float(r['KW'] or 0):.4f} {float(r['F1'] or 0):.4f} {float(r['CitFaith'] or 0):.4f} {float(r['CtxCov'] or 0):.4f} {r['Lat']:>6} {float(r['score'] or 0):.4f}")

best = max(rows, key=lambda r: float(r['score'] or 0))
print()
print(f"BEST score: exp#{best['exp_id']} {best['retriever']} k={best['k']} score={best['score']} KW={best['KW']} F1={best['F1']} CitFaith={best['CitFaith']} CtxCov={best['CtxCov']}")
print()
print("Top 5 by score:")
top5 = sorted(rows, key=lambda r: float(r['score'] or 0), reverse=True)[:5]
for r in top5:
    print(f"  exp#{r['exp_id']:>2} {r['retriever']:<18} k={r['k']} score={r['score']} KW={r['KW']} F1={r['F1']} note={r['note'][:60]}")
