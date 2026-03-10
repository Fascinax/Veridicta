import requests

ES_URL = "https://legimonaco.mc/~~search/depot/_search"

LABOUR_THEMATICS = [
    "social_protection_sociale",
    "social_conditions_de_travail",
    "social_contrats_de_travail",
    "social_rupture_du_contrat_de_travail",
    "social_securite_au_travail",
    "social_relations_collectives_du_travail",
    "social_chomage_et_reclassement",
    "social_travailleurs_etrangers",
    "social_apprentissage_et_formation_professionnelle",
    "social_contentieux",
    "social_social_general",
]

LABOUR_JURISDICTIONS = [
    "tribunal-travail",
    "cour-appel",
    "cour-revision",
    "tribunal-supreme",
    "tribunal-premiere-instance",
    "cour-superieure-arbitrage",
]


def count(query):
    r = requests.post(ES_URL, json={"size": 0, "track_total_hits": True, "query": query}, timeout=30)
    r.raise_for_status()
    return r.json()["hits"]["total"]["value"]


# 1. Regulation with labour thematics
n_reg = count({"bool": {"must": [
    {"terms": {"type": ["regulation"]}},
    {"terms": {"thematic": LABOUR_THEMATICS}},
]}})
print(f"regulation + labour thematics: {n_reg}")

# 2. Cases from non-travail courts with labour thematics
n_cases = count({"bool": {"must": [
    {"terms": {"type": ["case"]}},
    {"terms": {"jurisdiction": LABOUR_JURISDICTIONS}},
    {"terms": {"thematic": LABOUR_THEMATICS}},
]}})
print(f"case (all labour courts) + labour thematics: {n_cases}")

# 3. Already scraped from tribunal-travail only
n_travail = count({"bool": {"must": [
    {"terms": {"type": ["case"]}},
    {"term": {"jurisdiction": "tribunal-travail"}},
]}})
print(f"case tribunal-travail only (baseline): {n_travail}")

# 4. legislation (already scraped)
n_leg = count({"bool": {"must": [
    {"terms": {"type": ["legislation"]}},
    {"terms": {"thematic": LABOUR_THEMATICS}},
]}})
print(f"legislation + labour thematics (baseline): {n_leg}")

# 5. regulation with labour thematics broken down by nature
r5 = requests.post(ES_URL, json={
    "size": 0,
    "query": {"bool": {"must": [
        {"terms": {"type": ["regulation"]}},
        {"terms": {"thematic": LABOUR_THEMATICS}},
    ]}},
    "aggs": {"natures": {"terms": {"field": "tncNature", "size": 20}}},
}, timeout=30)
print("\nRegulation sub-types with labour thematics:")
for b in r5.json()["aggregations"]["natures"]["buckets"]:
    print(f"  {b['key']}: {b['doc_count']}")

# 6. TAI documents
n_tai = count({"terms": {"type": ["tai"]}})
print(f"\ntai total: {n_tai}")
