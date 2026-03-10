import requests, json

ES_URL = "https://legimonaco.mc/~~search/depot/_search"

r = requests.post(ES_URL, json={
    "size": 0,
    "aggs": {
        "types": {"terms": {"field": "type", "size": 50}},
        "jurisdictions": {"terms": {"field": "jurisdiction", "size": 20}},
        "thematics": {"terms": {"field": "thematic", "size": 50}},
        "natures": {"terms": {"field": "tncNature", "size": 30}},
    }
}, timeout=30)
r.raise_for_status()
data = r.json()

print("=== Document types ===")
for b in data["aggregations"]["types"]["buckets"]:
    print(f"  {b['key']}: {b['doc_count']}")

print("\n=== Jurisdictions ===")
for b in data["aggregations"]["jurisdictions"]["buckets"]:
    print(f"  {b['key']}: {b['doc_count']}")

print("\n=== Top thematics (all) ===")
for b in data["aggregations"]["thematics"]["buckets"]:
    print(f"  {b['key']}: {b['doc_count']}")

print("\n=== tncNature (legislation sub-types) ===")
for b in data["aggregations"]["natures"]["buckets"]:
    print(f"  {b['key']}: {b['doc_count']}")
