# Stage 0 - Protocole experimental executable

## But

Ce protocole execute l'etape 0 definie dans la RFC RAG : localiser ou se perd le bon passage entre retrieval, ranking, injection et generation.

Le protocole est volontairement court et operatoire. Il produit cinq artefacts :

1. un sous-ensemble de 40 questions difficiles
2. deux runs complets comparables sur ce sous-ensemble
3. deux runs retrieval-only comparables sur ce sous-ensemble
4. deux snapshots top-20 de retrieval
5. un paquet d'annotation manuelle pour classer les erreurs

## Question de decision

A la fin de ce protocole, on doit pouvoir trancher entre trois lectures :

- le bon passage est absent du pool pertinent
- le bon passage est present mais mal classe
- le bon passage est injecte mais mal exploite ou mal mesure

## Configurations a comparer

Le protocole compare volontairement deux configurations de reference deja connues :

- config A, haut recall : `lancedb_graph`, `k=5`, `prompt_version=3`, `backend=copilot`, `model=gpt-4.1`
- config B, haut F1 : `hybrid_graph`, `k=5`, `prompt_version=3`, `backend=copilot`, `model=gpt-4.1`

Ces deux configurations servent de contraste : l'une priorise mieux le rappel, l'autre la qualite de reponse.

## Preconditions

- se placer a la racine du repo
- avoir deja les index construits
- avoir GitHub Copilot fonctionnel pour les runs complets
- garder Neo4j disponible pour `hybrid_graph` et `lancedb_graph`

## Arborescence de travail

Tous les artefacts Stage 0 seront ranges ici :

- `eval/test_questions_stage0_bottom40.json`
- `eval/results/stage0/`
- `data/audit/stage0/`
- `eval/results/stage0/annotation_packet.jsonl`

## Etape 1 - Creer le sous-ensemble des 40 pires questions

Cette etape part du run haut recall existant et extrait les 40 plus mauvaises questions par `word_f1`.

Commande PowerShell :

```powershell
New-Item -ItemType Directory -Force eval/results/stage0 | Out-Null
New-Item -ItemType Directory -Force data/audit/stage0 | Out-Null
python -c 'import json; from pathlib import Path; src = Path("eval/results/copilot-lancedb-graph/eval_20260311_003251.jsonl"); rows = [json.loads(line) for line in src.open(encoding="utf-8")]; worst_ids = {row["question_id"] for row in sorted(rows, key=lambda row: row["word_f1"] if row["word_f1"] is not None else 999)[:40]}; questions = json.load(Path("eval/test_questions.json").open(encoding="utf-8")); subset = [question for question in questions if question["id"] in worst_ids]; Path("eval/test_questions_stage0_bottom40.json").write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")'
```

Sortie attendue :

- le fichier `eval/test_questions_stage0_bottom40.json` existe
- il contient 40 questions

Verification rapide :

```powershell
python -c 'import json; from pathlib import Path; subset = json.load(Path("eval/test_questions_stage0_bottom40.json").open(encoding="utf-8")); print(len(subset))'
```

## Etape 2 - Lancer les deux runs complets sur le sous-ensemble

Objectif : comparer la reponse finale et l'injection reelle de contexte sur le meme lot difficile.

### Run A - `lancedb_graph`

```powershell
$env:VERIDICTA_AUDIT_DIR = "data/audit/stage0/lancedb_graph"
$env:VERIDICTA_AUDIT_INCLUDE_CONTENT = "true"
python -m eval.evaluate --questions eval/test_questions_stage0_bottom40.json --retriever lancedb_graph --k 5 --backend copilot --model gpt-4.1 --prompt-version 3 --out eval/results/stage0/lancedb_graph_full
```

### Run B - `hybrid_graph`

```powershell
$env:VERIDICTA_AUDIT_DIR = "data/audit/stage0/hybrid_graph"
$env:VERIDICTA_AUDIT_INCLUDE_CONTENT = "true"
python -m eval.evaluate --questions eval/test_questions_stage0_bottom40.json --retriever hybrid_graph --k 5 --backend copilot --model gpt-4.1 --prompt-version 3 --out eval/results/stage0/hybrid_graph_full
```

Sorties attendues :

- un fichier `eval_*.jsonl` dans chaque dossier de `eval/results/stage0/*_full/`
- un fichier `data/audit/stage0/*/queries.jsonl` pour chaque config

## Etape 3 - Lancer les deux runs retrieval-only sur le meme sous-ensemble

Objectif : isoler la couche retrieval du comportement du LLM.

### Run A - `lancedb_graph`, retrieval-only

```powershell
python -m eval.evaluate --questions eval/test_questions_stage0_bottom40.json --retriever lancedb_graph --k 5 --retrieval-only --prompt-version 3 --out eval/results/stage0/lancedb_graph_retrieval_only
```

### Run B - `hybrid_graph`, retrieval-only

```powershell
python -m eval.evaluate --questions eval/test_questions_stage0_bottom40.json --retriever hybrid_graph --k 5 --retrieval-only --prompt-version 3 --out eval/results/stage0/hybrid_graph_retrieval_only
```

Lecture attendue :

- si le retrieval-only est deja faible sur les memes questions, la perte vient tot
- si le retrieval-only est correct mais le full run chute, la perte se situe plutot dans ranking fin, injection ou generation

## Etape 4 - Exporter le top 20 de retrieval pour chaque question

Objectif : distinguer proprement `retrieval_absent` de `ranking_bad_order`.

### Snapshot A - `lancedb_graph`, top 20

```powershell
python -c 'import json; from pathlib import Path; from types import SimpleNamespace; from eval.evaluate import _collect_retrievals, _load_optional_retrievers, _load_embedder, load_index, load_questions; questions = load_questions(Path("eval/test_questions_stage0_bottom40.json")); index, chunks = load_index(Path("data/index")); embedder = _load_embedder(); args = SimpleNamespace(retriever="lancedb_graph"); bm25, neo4j_mgr, lancedb_table = _load_optional_retrievers(args, Path("data/index")); retrieved_all = _collect_retrievals(questions, index, chunks, embedder, k=20, bm25=bm25, neo4j_mgr=neo4j_mgr, lancedb_table=lancedb_table); out = Path("eval/results/stage0/lancedb_graph_top20.jsonl"); out.write_text("\n".join(json.dumps({"question_id": q.id, "question": q.question, "chunks": rows}, ensure_ascii=False) for q, rows in zip(questions, retrieved_all)) + "\n", encoding="utf-8")'
```

### Snapshot B - `hybrid_graph`, top 20

```powershell
python -c 'import json; from pathlib import Path; from types import SimpleNamespace; from eval.evaluate import _collect_retrievals, _load_optional_retrievers, _load_embedder, load_index, load_questions; questions = load_questions(Path("eval/test_questions_stage0_bottom40.json")); index, chunks = load_index(Path("data/index")); embedder = _load_embedder(); args = SimpleNamespace(retriever="hybrid_graph"); bm25, neo4j_mgr, lancedb_table = _load_optional_retrievers(args, Path("data/index")); retrieved_all = _collect_retrievals(questions, index, chunks, embedder, k=20, bm25=bm25, neo4j_mgr=neo4j_mgr, lancedb_table=lancedb_table); out = Path("eval/results/stage0/hybrid_graph_top20.jsonl"); out.write_text("\n".join(json.dumps({"question_id": q.id, "question": q.question, "chunks": rows}, ensure_ascii=False) for q, rows in zip(questions, retrieved_all)) + "\n", encoding="utf-8")'
```

Sorties attendues :

- `eval/results/stage0/lancedb_graph_top20.jsonl`
- `eval/results/stage0/hybrid_graph_top20.jsonl`

Lecture attendue :

- si le bon passage n'est pas dans ces snapshots top-20, la perte est en amont du ranking final
- s'il y est mais n'entre pas dans les chunks injectes, la perte est surtout de ranking ou de fenetrage d'injection

## Etape 5 - Construire le paquet d'annotation manuelle

Objectif : produire un fichier unique qui rassemble question, reference, prediction, et chunks injectes pour annotation humaine.

Commande PowerShell :

```powershell
python -c 'import json; from pathlib import Path; questions = {q["id"]: q for q in json.load(Path("eval/test_questions_stage0_bottom40.json").open(encoding="utf-8"))}; questions_by_text = {q["question"]: q["id"] for q in questions.values()}; chunks = {}; \
for line in Path("data/processed/chunks.jsonl").open(encoding="utf-8"): \
    row = json.loads(line); chunks[row["chunk_id"]] = {"chunk_id": row.get("chunk_id"), "doc_id": row.get("doc_id"), "titre": row.get("titre", row.get("title", "")), "text": row.get("text", ""), "date": row.get("date", ""), "type": row.get("type", "")}; \
results_path = sorted(Path("eval/results/stage0/lancedb_graph_full").glob("eval_*.jsonl"))[-1]; results = {row["question_id"]: row for row in (json.loads(line) for line in results_path.open(encoding="utf-8"))}; \
top20 = {row["question_id"]: row["chunks"] for row in (json.loads(line) for line in Path("eval/results/stage0/lancedb_graph_top20.jsonl").open(encoding="utf-8"))}; \
audit_path = Path("data/audit/stage0/lancedb_graph/queries.jsonl"); packet = []; \
for line in audit_path.open(encoding="utf-8"): \
    audit = json.loads(line); matched = questions_by_text.get(audit["query"].get("text", "")); \
    if not matched: continue; \
    used_chunks = []; \
    for chunk_meta in audit["retrieval"]["chunks"]: \
        if not chunk_meta.get("used_in_prompt"): continue; \
        chunk_id = chunk_meta.get("chunk_id"); merged = dict(chunk_meta); merged["text"] = chunks.get(chunk_id, {}).get("text", ""); used_chunks.append(merged); \
    packet.append({"question_id": matched, "question": questions[matched]["question"], "reference_answer": questions[matched]["reference_answer"], "reference_keywords": questions[matched]["reference_keywords"], "prediction": results[matched]["answer"], "word_f1": results[matched]["word_f1"], "keyword_recall": results[matched]["keyword_recall"], "citation_faithfulness": results[matched]["citation_faithfulness"], "context_coverage": results[matched]["context_coverage"], "top20_chunks": top20.get(matched, []), "used_chunks": used_chunks}); \
out = Path("eval/results/stage0/annotation_packet.jsonl"); out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in packet) + "\n", encoding="utf-8")'
```

Sortie attendue :

- `eval/results/stage0/annotation_packet.jsonl`

Usage :

- annoter manuellement 30 a 50 cas a partir de ce fichier
- pour chaque question, classer l'erreur dans une seule categorie primaire

Taxonomie minimale :

- `retrieval_absent`
- `retrieval_present_wrong_passage`
- `ranking_bad_order`
- `injected_but_bad_use`
- `semantically_ok_but_metric_penalty`

## Etape 6 - Test prompt a contexte constant

Objectif : verifier si le prompt seul deplace fortement le F1 a contexte fixe.

Selection :

- prendre 10 cas du paquet d'annotation ou les 5 chunks injectes semblent globalement pertinents

Procedure :

1. copier ces 10 cas dans un fichier temporaire local
2. pour chaque cas, rejouer la generation avec exactement les memes `used_chunks`
3. comparer `prompt_version=1`, `prompt_version=2`, `prompt_version=3`

Script minimal a executer localement si besoin :

```python
import json
from pathlib import Path

from retrievers.baseline_rag import answer
from eval.evaluate import word_f1

packet_path = Path("eval/results/stage0/prompt_ablation_sample.jsonl")
rows = [json.loads(line) for line in packet_path.open(encoding="utf-8")]

for row in rows:
    print(f"\n## {row['question_id']} - {row['question']}")
    used_chunks = row["used_chunks"]
    for prompt_version in (1, 2, 3):
        prediction = answer(
            row["question"],
            used_chunks,
            backend="copilot",
            model="gpt-4.1",
            prompt_version=prompt_version,
        )
        score = word_f1(prediction, row["reference_answer"])
        print(f"prompt_v{prompt_version}: F1={score:.4f}")
```

Lecture attendue :

- si le F1 varie fortement a contexte constant, la couche generation est impliquée
- si le F1 bouge peu, la perte est probablement en amont

## Etape 7 - Decision rules

### Conclure `ranking prioritaire` si

- le bon passage apparait souvent dans les chunks recuperes
- mais trop bas ou trop rarement dans les meilleurs chunks injectes
- et un reranker plus fort ou un pool plus large semble le faire remonter

### Conclure `chunking prioritaire` si

- les chunks injectes sont regulierement trop larges, heterogenes ou hors cible
- ou si le bon document apparait mais pas le bon passage exploitable

### Conclure `generation prioritaire` si

- les chunks injectes sont juges globalement suffisants
- mais le test prompt a contexte constant deplace fortement le F1

### Conclure `metrique a revisiter` si

- plusieurs cas sont juges juridiquement corrects et bien sources
- mais restent fortement penalises au Word F1 pour des raisons de paraphrase ou de structure

## Definition of done

Le protocole est termine quand les cinq conditions suivantes sont remplies :

1. les 4 runs Stage 0 existent sur disque
2. les 2 snapshots top-20 existent sur disque
3. `annotation_packet.jsonl` existe
4. 30 a 50 erreurs ont ete annotees manuellement
5. une conclusion ecrite tranche entre perte majoritaire en retrieval, ranking, injection ou generation

## Resultat attendu

Le resultat attendu n'est pas une nouvelle techno. Le resultat attendu est une preuve exploitable pour decider si la prochaine iteration doit commencer par le reranker, le chunking ou la couche de generation.