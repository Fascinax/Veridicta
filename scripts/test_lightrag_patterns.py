"""Quick test for LightRAG article-extraction regex patterns."""
import re
import jsonlines

# Article-level citations in jurisprudence/regulation
# Matches: "l'article 5 de la loi n° 729", "l'article 1er de l'ordonnance souveraine n° 3.162"
ARTICLE_CITE_RE = re.compile(
    r"(?:l['\u2019\s]|les\s+)?articles?\s+"
    r"(\d+(?:-\d+)?(?:er|bis|ter)?)"
    r"(?:[^,\n]{0,30})?"
    r"\s+(?:de\s+la\s+|du\s+|de\s+l['\u2019]\s*)"
    r"(loi\s+n[o\u00b0.]\s*[\d][\d.\-]*"
    r"|ordonnance\s+(?:souveraine\s+)?n[o\u00b0.]\s*[\d][\d.\-]*"
    r"|code\s+du\s+travail)",
    re.IGNORECASE,
)

# Article headers in legislation/regulation text
# Matches standalone "Art. 5 -", "Article 1er.", "Art. 15."
ARTICLE_HEADER_RE = re.compile(
    r"(?:^|\n)\s*Art(?:icle)?\.?\s+(\d+(?:-\d+)?(?:er|bis|ter)?)\s*[-–—\.]",
    re.IGNORECASE | re.MULTILINE,
)

# MODIFIE / ABROGE: extract law reference being modified/repealed
MODIFIE_RE = re.compile(
    r"(?:modifiant|portant\s+modification\s+(?:de\s+(?:la\s+)?)?|abrogeant|portant\s+abrogation\s+de\s+)"
    r"(?:certaines\s+dispositions\s+de\s+la\s+|la\s+|l['\u2019]\s*|du\s+)?"
    r"(loi\s+n[o\u00b0.]\s*[\d][\d.\-]*|ordonnance\s+(?:souveraine\s+)?n[o\u00b0.]\s*[\d][\d.\-]*)",
    re.IGNORECASE,
)

# "Voir l'article X de la loi n° Y" (LegiMonaco cross-ref markers)
VOIR_ARTICLE_RE = re.compile(
    r"voir\s+l['\u2019]\s*article\s+(\d+(?:-\d+)?(?:er|bis|ter)?)"
    r"\s+de\s+(?:la\s+|l['\u2019]\s*)?"
    r"(loi\s+n[o\u00b0.]\s*[\d][\d.\-]*|ordonnance\s+(?:souveraine\s+)?n[o\u00b0.]\s*[\d][\d.\-]*)",
    re.IGNORECASE,
)

def test_patterns():
    cite_matches = []
    header_matches = []
    modifie_matches = []
    voir_matches = []

    n_jur = 0
    n_leg = 0

    with jsonlines.open("data/processed/chunks.jsonl") as reader:
        for chunk in reader:
            text = chunk.get("text", "")
            ctype = chunk.get("type", "")

            if ctype == "jurisprudence" and n_jur < 500:
                n_jur += 1
                for m in ARTICLE_CITE_RE.finditer(text):
                    cite_matches.append((m.group(1), m.group(2), chunk["doc_id"][:12]))

            elif ctype in ("legislation", "regulation") and n_leg < 500:
                n_leg += 1
                for m in ARTICLE_HEADER_RE.finditer(text):
                    header_matches.append((m.group(1), chunk["doc_id"][:12], chunk.get("titre", "")[:40]))
                for m in MODIFIE_RE.finditer(text):
                    modifie_matches.append((m.group(1), chunk["doc_id"][:12], chunk.get("titre", "")[:40]))
                for m in VOIR_ARTICLE_RE.finditer(text):
                    voir_matches.append((m.group(1), m.group(2), chunk["doc_id"][:12]))

    print(f"\n=== ARTICLE CITATIONS in jurisprudence (out of {n_jur} chunks) ===")
    for art, law, did in cite_matches[:12]:
        print(f"  art={art!r:12} law={law!r:42} doc={did}")
    print(f"  Total: {len(cite_matches)}")

    print(f"\n=== ARTICLE HEADERS in legislation/regulation (out of {n_leg} chunks) ===")
    for art, did, titre in header_matches[:12]:
        print(f"  Art.{art!r:8} doc={did} titre={titre}")
    print(f"  Total: {len(header_matches)}")

    print(f"\n=== MODIFIE/ABROGE in legislation/regulation ===")
    for law, did, titre in modifie_matches[:12]:
        print(f"  law={law!r:42} doc={did} titre={titre}")
    print(f"  Total: {len(modifie_matches)}")

    print(f"\n=== VOIR ARTICLE in legislation/regulation ===")
    for art, law, did in voir_matches[:12]:
        print(f"  art={art!r:12} law={law!r:42} doc={did}")
    print(f"  Total: {len(voir_matches)}")


if __name__ == "__main__":
    test_patterns()
