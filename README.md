# Veridicta

> **Tagline**: Retrieval‑Augmented Generative AI for reliable, explainable French legal answers.

---

## 1. Vision

Construire un agent conversationnel juridique complet qui combine **LightRAG**, **PathRAG**, **QLoRA fine‑tuned LLM**, **Tool‑Calling** et **Guardrails** afin de délivrer des réponses précises, rapides et traçables à des questions de droit français.

## 2. Objectifs quantifiés (KPIs)

| Indicateur                      | Cible                   |
| ------------------------------- | ----------------------- |
| Latence p95                     | < 1 s / 256 tokens      |
| Coût variable                   | < 0,005 € / 1000 tokens |
| Hallucinations                  | < 5 % (set de 50 Q/R)   |
| Exact Match (LegalBench subset) | ≥ 70 %                  |

## 3. Stack technologique

* **Python 3.11**
* Embeddings : `sentence‑transformers` (MiniLM‑FR)
* Retrieval : **FAISS**, **LightRAG**, **PathRAG**
* LLM : **Mistral‑7B** fine‑tune QLoRA 4‑bit (ou Mixtral 8×7B)
* Orchestration : **LangChain** + **vLLM**
* Knowledge Graph : **Neo4j**
* Guardrails : **Aporia 2024**, **LlamaGuard**
* UI : **Streamlit**
* Ops/Bench : **wandb**, **pytest**

## 4. Arborescence du dépôt

```
<PROJECT_NAME>/
├── data_ingest/         # Scrapers + parsers Legifrance, JORF
├── retrievers/          # FAISS, LightRAG, PathRAG impl.
├── models/              # Checkpoints, QLoRA configs, prompts
├── tools/               # API calcul indemnités, date diff, etc.
├── eval/                # evaluate.py, question sets, notebooks
├── ui/                  # Streamlit app
└── README.md            # ce fichier
```

## 5. Planning (8 semaines)

| Phase            | Semaine | Livrable clé                     |
| ---------------- | ------- | -------------------------------- |
| 0. Kick‑off      | 0       | Repo, README, métriques cibles   |
| 1. Corpus        | 1       | JSONL du Code du Travail & Civil |
| 2. Baseline RAG  | 2       | FAISS + prompt basique           |
| 3. LightRAG      | 3       | Bench latence ×2                 |
| 4. PathRAG       | 4       | Bench tokens ‑30 %               |
| 5. QLoRA + Tools | 5‑6     | Hallucinations ‑40 %             |
| 6. Guardrails    | 7       | Dashboard wandb                  |
| 7. Demo UI       | 8       | Vidéo + slides                   |

## 6. Installation rapide

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 7. Utilisation

```bash
python data_ingest/legifrance_scraper.py
python retrievers/build_faiss.py
streamlit run ui/app.py
```

## 8. Évaluation

Lancer :

```bash
python eval/evaluate.py --model mistral7B_qlora --retriever baseline
```

Rapporte latence, EM, hallucinations, coût.

## 9. Licence

À définir (par défaut MIT pour le code, CC‑BY pour les datasets publics).

---

## 10. Sources de données monégasques

Voici la **short-list des « end-points » monégasques à intégrer dans le pipeline d'ingestion** ; ce sont les pages racines (ou motifs d'URL) qui donnent ≈ 100 % du droit positif et de la jurisprudence publiés :

| Catégorie                                                              | Pour quoi faire ?                                                                                  | Page                                                                                                                     | Format dominant | Points d'attention                                                                                                                                                                                                                                 |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Bulletin officiel – Journal de Monaco**                              | Toutes les lois, ordonnances souveraines, arrêtés ministériels, avis, etc.                         | `https://journaldemonaco.gouv.mc/` → lien **« Tous les journaux »** (liste paginée par année ; chaque numéro a son PDF) ([Journal de Monaco][1]) | PDF hebdo       | ­• Scraper la page d'index pour récupérer `numéro`, `date`, `pdf_url`.<br>• Le nom de fichier suit `JM_numero_date.pdf`; garde-le comme ID unique.<br>• Vérifie le robots.txt : l'archivage institutionnel est autorisé pour usage non commercial. |
| **Codes monégasques consolidés**                                       | Textes codifiés (civil, pénal, proc. civile/pénale, environnement, mer, international privé, etc.) | `https://legimonaco.mc/` → menu **Codes** (liste + navigation HTML par livre/titre) ([Legimonaco][2], [Legimonaco][3])                           | HTML + PDF      | ­• Chaque code = un arbre `<article>` avec ancre `id="art_<num>"` : parfait pour le découpage chunk=article.<br>• Pense à historiser : les mises à jour créent un nouvel encart **« version en vigueur au … »**.                                   |
| **Textes législatifs & réglementaires isolés**                         | Lois non codifiées, ordonnances, décrets abrogeant/complétant, traités                             | `https://legimonaco.mc/` → menu **Textes législatifs** & **Textes réglementaires** (filtres par année, type) ([Legimonaco][2])                   | HTML + PDF      | ­• Les URLs contiennent `/legis/` ou `/reglementaire/` + ID numérique.<br>• Extrais aussi le champ « Abroge » pour alimenter les arêtes du graphe (relation *modifie*).                                                                  |
| **Jurisprudence interne (Tribunal – Cour d'Appel – Cour de Révision)** | Décisions monégasques signées                                                                      | `https://legimonaco.mc/jurisprudence/…` (filtres par juridiction et date) ([Legimonaco][4])                                                      | HTML            | ­• Structure constante : `<h1>` pour la juridiction, tableau méta (date, partie, visa), puis corps en `<p>` numérotés.<br>• Ajoute un champ `formation` (*chambre civile*, etc.) si présent ; utile pour la pertinence.                            |
| **Jurisprudence bilingue**                                | Décisions non encore migrées vers LegiMonaco ; doublons utiles pour OCR                            | `https://juricaf.org/recherche/?facet_pays:Monaco` (filtré par juridiction) ([Juricaf][5])                                                       | HTML            | ­• Juricaf garde des PDFs scannés des années 1950-2000 que LegiMonaco ne contient pas encore.                                                                                                                                                      |
| **Travaux parlementaires**                                             | Exposés des motifs, rapports sur projets  de loi                                     | `https://legimonaco.mc/~~write/explorer/type%3DlegislativeWork` (ou filtre « Travaux législatifs ») ([Legimonaco][6])                            | PDF             | ­• Chaque entrée fournit le **numéro de projet** utile pour relier aux lois promulgées.                                                                                                                                        |

### 🏗️ Conseils d'implémentation rapide

1. **Scraper HTML léger** : `requests` + `BeautifulSoup` suffit, sauf pour le Journal de Monaco qui charge l'index via JS ; utiliser `playwright` headless pour récupérer la liste des bulletins, puis basculer en download direct des PDFs.
2. **Détection d'entités monégasques** : ajouter des motifs `Loi n° \d+`, `Ordonnance Souveraine n° \d+`, `Arrêté Ministériel n° \d+`, `Projet de loi n° \d+`.
3. **Versioning** : pour les codes, conserver le marqueur « en vigueur au … ». Stocker la date d'entrée en vigueur comme propriété du nœud *Article*.
4. **Delta-fetch** :
   * Journal : un nouveau PDF chaque vendredi (numéro séquentiel) → cron hebdo.
   * Legimonaco : la page d'accueil indique le **dernier journal publié** (numéro + date) ([Legimonaco][2]) ; s'en servir comme check rapide.
5. **Graph builder** : relier :
   * *Texte* → *modifie*/*abroge* → *Texte cible*
   * *Décision* → *cite* → *Article*  → *about* → *Texte promulgué*
6. **Langue** : tout est en français ; garder un pipeline d'embedding FR (CamemBERT-based).

[1]: https://journaldemonaco.gouv.mc/ "Journal de Monaco: Accueil"
[2]: https://legimonaco.mc/ "Legimonaco"
[3]: https://legimonaco.mc/~~write/explorer/type%3Dtc/titleAsc "Codes - Legimonaco"
[4]: https://legimonaco.mc/jurisprudence/cour-revision/2024/03-18-30454 "Cour de révision, 18 mars 2024, m. c. A. et autres c - Legimonaco"
[5]: https://juricaf.org/recherche/%2B/facet_pays%3AMonaco%2Cfacet_pays_juridiction%3AMonaco_%7C_Cour_de_r%C3%A9vision "La jurisprudences de Monaco | Cour de révision - Juricaf"
[6]: https://legimonaco.mc/~~write/explorer/type%3DlegislativeWork/dateDesc "83 documents dans \"Travaux législatifs\" - Legimonaco"

---

*Dernière mise à jour : 2025-05-10*
