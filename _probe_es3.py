import requests, json

ES_URL = "https://legimonaco.mc/~~search/depot/_search"
LABOUR_THEMATICS = [
    "social_protection_sociale", "social_conditions_de_travail",
    "social_contrats_de_travail", "social_rupture_du_contrat_de_travail",
    "social_securite_au_travail", "social_relations_collectives_du_travail",
    "social_chomage_et_reclassement", "social_travailleurs_etrangers",
    "social_apprentissage_et_formation_professionnelle", "social_contentieux", "social_social_general",
]

# Sample one regulation doc to see all fields
r = requests.post(ES_URL, json={
    "size": 1,
    "query": {"bool": {"must": [
        {"terms": {"type": ["regulation"]}},
        {"terms": {"thematic": LABOUR_THEMATICS}},
    ]}},
    "sort": [{"date": "desc"}],
}, timeout=30)
r.raise_for_status()
hit = r.json()["hits"]["hits"][0]["_source"]
print("=== Regulation doc fields ===")
for k, v in hit.items():
    val = str(v)[:120] if not isinstance(v, list) else str(v)[:120]
    print(f"  {k}: {val}")

# Sample cross-court case
r2 = requests.post(ES_URL, json={
    "size": 1,
    "query": {"bool": {"must": [
        {"terms": {"type": ["case"]}},
        {"terms": {"jurisdiction": ["cour-appel", "cour-revision"]}},
        {"terms": {"thematic": LABOUR_THEMATICS}},
    ]}},
    "sort": [{"date": "desc"}],
}, timeout=30)
r2.raise_for_status()
hit2 = r2.json()["hits"]["hits"][0]["_source"]
print("\n=== Cross-court case fields ===")
for k, v in hit2.items():
    val = str(v)[:120] if not isinstance(v, list) else str(v)[:120]
    print(f"  {k}: {val}")
