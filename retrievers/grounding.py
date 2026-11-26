"""Claim-level grounding checks and safe abstention for generated answers."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from retrievers.traceability import normalize_citations

_SOURCE_PATTERN = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)
_TOKEN_PATTERN = re.compile(r"[a-zàâäéèêëîïôöùûüÿçœ]{4,}", re.IGNORECASE)
_SHORT_TOKEN_PATTERN = re.compile(r"[a-z]{2,}", re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*")
_NEGATION_PATTERN = re.compile(
    r"(?:"
    r"\b(?:ne\s+|n['’])(?:[^\n.!?;:,]{0,50}?)\b(?:pas(?:\s+non\s+plus)?|jamais|plus|point|gu[eè]re|rien|personne|aucun\w*|nul(?:le|s|les|lement)?)\b"
    r"|\b(?:ne\s+|n['’])(?:[^\n.!?;:,]{0,50}?)\bni\b(?:[^\n.!?;:]{0,60}?\bni\b)*"
    r"|\b(?<!\bla\s)(?<!\bune\s)(?<!\bcette\s)(?<!\btoute\s)(?<!\bchaque\s)(?:personne|rien)\b[^\n.!?;:,]{0,30}?\b(?:ne\s+|n['’])(?![^\n.!?;:,]{0,40}?\b(?:pas|jamais|plus|point|gu[eè]re|aucun\w*|nul(?:le|s|les|lement)?)\b)(?:peut|peuvent|pourra(?:i(?:t|ent)|nt)?|puisse(?:nt)?|saurai(?:t|ent)|doit|doivent|devra(?:i(?:t|ent)|nt)?|cesse(?:nt)?|ose(?:nt)?)?"
    r"|\b(?:nul(?:le|s|les)?|aucun\w*)\b(?:[^\n.!?;:,]{0,30}?\b(?:ne\s+|n['’])(?![^\n.!?;:,]{0,40}?\b(?:pas|jamais|plus|point|gu[eè]re|rien|personne)\b)(?:peut|peuvent|pourra(?:i(?:t|ent)|nt)?|puisse(?:nt)?|saurai(?:t|ent)|doit|doivent|devra(?:i(?:t|ent)|nt)?|cesse(?:nt)?|ose(?:nt)?))?"
    r"|\b(?:ne\s+|n['’])(?:peut|peuvent|pourra(?:i(?:t|ent)|nt)?|puisse(?:nt)?|saurai(?:t|ent)|doit|doivent|devra(?:i(?:t|ent)|nt)?|cesse(?:nt)?|ose(?:nt)?)\b"
    r"|\bni\b(?:[^\n.!?;:]{0,60}?\bni\b)*"
    r"|\bnon\b(?!\s+(?:seulement|plus)\b)"
    r"|\b(?:pas|jamais|sans|interdit\w*|interdir\w*|exclu\w*|nullement|prohib\w*|inapplicable\w*)\b"
    r")",
    re.IGNORECASE,
)
DEFAULT_MINIMUM_SUPPORT = 0.65
_PARTIAL_ABSTENTION_NOTICE = (
    "Certaines affirmations ont été écartées : les sources fournies "
    "ne permettent pas de les confirmer."
)
_EMPTY_ABSTENTION_LEAD = (
    "Les sources fournies ne permettent pas de confirmer cette réponse."
)
_EMPTY_ABSTENTION_TAIL = (
    "Je m'abstiens de présenter les éléments insuffisamment étayés."
)
_SYSTEM_ABSTENTION_NOTICES = frozenset(
    {
        " ".join(_PARTIAL_ABSTENTION_NOTICE.casefold().split()),
        " ".join(_EMPTY_ABSTENTION_LEAD.casefold().split()),
        " ".join(_EMPTY_ABSTENTION_TAIL.casefold().split()),
        " ".join(f"{_EMPTY_ABSTENTION_LEAD} {_EMPTY_ABSTENTION_TAIL}".casefold().split()),
    }
)
_ABSTENTION_MARKERS = (
    "les sources fournies ne permettent",
    "les sources ne permettent",
    "les sources fournies ne précisent",
    "élément non précisé par les sources",
    "je ne peux pas confirmer",
    "information insuffisante",
)
_STOPWORDS = frozenset(
    {
        "avec",
        "ainsi",
        "dans",
        "des",
        "elle",
        "elles",
        "entre",
        "être",
        "leurs",
        "mais",
        "même",
        "notamment",
        "nous",
        "pour",
        "sans",
        "sont",
        "sur",
        "tous",
        "tout",
    }
)
_SHORT_STOPWORDS = frozenset(
    {
        "au",
        "aux",
        "ce",
        "de",
        "des",
        "du",
        "en",
        "est",
        "et",
        "il",
        "la",
        "le",
        "les",
        "ne",
        "on",
        "ou",
        "par",
        "que",
        "qui",
        "sa",
        "se",
        "ses",
        "son",
        "sur",
        "un",
        "une",
    }
)
_LEGAL_ABBREVIATIONS = frozenset(
    {
        "al",
        "art",
        "arts",
        "bull",
        "c",
        "cass",
        "cf",
        "chap",
        "civ",
        "com",
        "crim",
        "d",
        "dec",
        "déc",
        "dr",
        "ed",
        "éd",
        "etc",
        "ex",
        "l",
        "m",
        "me",
        "mlle",
        "mme",
        "mod",
        "modif",
        "n",
        "no",
        "nos",
        "obs",
        "ord",
        "p",
        "par",
        "para",
        "pen",
        "pén",
        "pp",
        "proc",
        "r",
        "ref",
        "réf",
        "sect",
        "soc",
        "st",
        "ste",
        "sup",
        "trav",
        "trib",
        "vol",
    }
)
_LEGAL_ACRONYM_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "cdi": ("contrat a duree indeterminee",),
    "cdd": ("contrat a duree determinee",),
    "sam": ("societe anonyme monegasque",),
    "sarl": ("societe a responsabilite limitee",),
    "snc": ("societe en nom collectif",),
    "scs": ("societe en commandite simple",),
    "sca": ("societe en commandite par actions",),
    "sci": ("societe civile immobiliere",),
    "ccss": ("caisse de compensation des services sociaux",),
    "car": ("caisse autonome des retraites",),
    "csm": ("caisses sociales de monaco",),
    "ccnt": ("convention collective nationale du travail",),
    "jom": ("journal de monaco",),
    "smic": ("salaire minimum interprofessionnel de croissance",),
    "smig": ("salaire minimum interprofessionnel garanti",),
    "tva": ("taxe sur la valeur ajoutee",),
    "rtt": ("reduction du temps de travail",),
    "cse": ("comite social et economique",),
    "dp": ("delegue du personnel", "delegues du personnel"),
    "os": ("ordonnance souveraine",),
    "am": ("arrete ministeriel",),
    "at": ("accident du travail", "accidents du travail"),
    "cp": ("conges payes", "conge paye", "code penal"),
    "ct": ("code du travail",),
    "cc": ("code civil", "convention collective"),
    "rgpd": ("reglement general sur la protection des donnees",),
}
_AMBIGUOUS_SHORT_ACRONYMS = frozenset({"am", "at", "car", "cc", "ce", "cp", "ct", "dp", "os"})
_UPPERCASE_NON_ACRONYMS = frozenset(
    {
        "ART",
        "ARTS",
        "AU",
        "AUX",
        "AVEC",
        "CES",
        "CET",
        "CHEZ",
        "CODE",
        "COUR",
        "DANS",
        "DE",
        "DES",
        "DONC",
        "DONT",
        "DU",
        "ELLE",
        "EN",
        "EST",
        "ET",
        "II",
        "III",
        "IL",
        "ILS",
        "IV",
        "IX",
        "JUGE",
        "LA",
        "LE",
        "LES",
        "LEUR",
        "LOI",
        "MAIS",
        "NE",
        "NI",
        "NON",
        "NOUS",
        "ON",
        "OU",
        "OUI",
        "PAR",
        "PAS",
        "POUR",
        "QUE",
        "QUI",
        "SA",
        "SANS",
        "SAUF",
        "SE",
        "SES",
        "SI",
        "SON",
        "SONT",
        "SOUS",
        "SUR",
        "TOUS",
        "TOUT",
        "UN",
        "UNE",
        "VERS",
        "VI",
        "VII",
        "VIII",
        "VOUS",
        "XI",
        "XII",
        "XIII",
        "XIV",
        "XIX",
        "XV",
        "XVI",
        "XVII",
        "XVIII",
        "XX",
    }
)
_ABBREV_BEFORE_DOT_PATTERN = re.compile(r"(?<![A-Za-zÀ-ÿ])([A-Za-zÀ-ÿ]+)$")
_DOTTED_ACRONYM_END_PATTERN = re.compile(r"(?:^|[^A-Za-zÀ-ÿ])(?:[A-Z]\.){2,}$")
_DOTTED_ACRONYM_PATTERN = re.compile(r"(?<![A-Za-zÀ-ÿ])(?:[A-Z]\.){2,}")
_UPPERCASE_ACRONYM_PATTERN = re.compile(r"(?<![A-Za-zÀ-ÿ])[A-Z]{2,6}(?![A-Za-zÀ-ÿ])")
_ALPHA_WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÿ]{2,}")
_SPAN_HEADING_PATTERN = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:(?:art(?:icle)?s?\.?\s+(?:[LRD]\.?\s*)?)?\d+(?:er|e|ème)?(?:-\d+)?(?:\s*(?:bis|ter|quater))?)\.$",
    re.IGNORECASE,
)
_ARTICLE_REF_BEFORE_DOT_PATTERN = re.compile(
    r"\bart(?:icle)?s?\.?\s+(?:[LRD]\.?\s*)?\d+(?:er|e|ème)?(?:-\d+)?(?:\s*(?:bis|ter|quater))?\.$",
    re.IGNORECASE,
)
_TRAILING_FIRST_CITATION = re.compile(r"[ \t]*\[Source[ \t]+\d+\]", re.IGNORECASE)
_TRAILING_NEXT_CITATION = re.compile(
    r"(?:[ \t]*[,;][ \t]*|[ \t]+(?:et|and)[ \t]+|[ \t]*)\[Source[ \t]+\d+\]",
    re.IGNORECASE,
)
_LOWERCASE_START = re.compile(r"[ \t]+[a-zàâäéèêëîïôöùûüÿçœ]")
_UPPERCASE_CONTINUATION = re.compile(r"[ \t]+[A-ZÀ-Ÿ]{2,}")
_CLAUSE_TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+'|vis-a-vis|[A-Za-z]+(?:-[A-Za-z]+)*|\d+(?:[.,]\d+)*|[,;:]"
)
_DETERMINERS = frozenset(
    {
        "auquel",
        "auxquelles",
        "auxquels",
        "ce",
        "certain",
        "certaine",
        "certaines",
        "certains",
        "ces",
        "cet",
        "cette",
        "chaque",
        "d'",
        "des",
        "desquelles",
        "desquels",
        "du",
        "duquel",
        "l'",
        "la",
        "ladite",
        "le",
        "ledit",
        "les",
        "lesdites",
        "lesdits",
        "leur",
        "leurs",
        "ma",
        "mes",
        "mon",
        "nos",
        "notre",
        "plusieurs",
        "quelque",
        "quelques",
        "sa",
        "ses",
        "son",
        "ta",
        "tel",
        "telle",
        "telles",
        "tels",
        "tes",
        "ton",
        "tous",
        "tout",
        "toute",
        "toutes",
        "un",
        "une",
        "vos",
        "votre",
    }
)
_PREPOSITION_STARTERS = frozenset(
    {
        "a",
        "apres",
        "au",
        "aupres",
        "aux",
        "avant",
        "avec",
        "chez",
        "concernant",
        "conformement",
        "contre",
        "d'",
        "dans",
        "de",
        "depuis",
        "des",
        "du",
        "durant",
        "en",
        "entre",
        "envers",
        "face",
        "grace",
        "hormis",
        "jusqu'",
        "jusque",
        "lors",
        "malgre",
        "moyennant",
        "outre",
        "par",
        "pendant",
        "pour",
        "quant",
        "relativement",
        "sans",
        "sauf",
        "selon",
        "sous",
        "suite",
        "suivant",
        "sur",
        "vers",
        "vis-a-vis",
    }
)
_CLAUSE_LEAD_CONJUNCTIONS = frozenset(
    {
        "ainsi",
        "alors",
        "car",
        "cependant",
        "comme",
        "donc",
        "enfin",
        "ensuite",
        "et",
        "lorsqu'",
        "lorsque",
        "mais",
        "neanmoins",
        "or",
        "ou",
        "puis",
        "puisqu'",
        "puisque",
        "qu'",
        "quand",
        "que",
        "quoique",
        "si",
        "tandis",
        "toutefois",
    }
)
_ETRE_FORMS = frozenset(
    {
        "est",
        "etaient",
        "etait",
        "etre",
        "furent",
        "fut",
        "sera",
        "seraient",
        "serait",
        "seront",
        "soient",
        "soit",
        "sont",
    }
)
_AVOIR_FORMS = frozenset(
    {
        "a",
        "aient",
        "ait",
        "aura",
        "auraient",
        "aurait",
        "auront",
        "avaient",
        "avait",
        "avoir",
        "eurent",
        "eut",
        "ont",
    }
)
_MODAL_FORMS = frozenset(
    {
        "devait",
        "devaient",
        "devra",
        "devraient",
        "devrait",
        "devront",
        "doit",
        "doive",
        "doivent",
        "peut",
        "peuvent",
        "pourra",
        "pourraient",
        "pourrait",
        "pourront",
        "pouvait",
        "pouvaient",
        "puisse",
        "puissent",
        "va",
        "veulent",
        "veut",
        "viennent",
        "vient",
        "vont",
    }
)
_PREVERBAL_CLITICS = frozenset({"lui", "s'", "se", "y"})
_PRONOUN_SUBJECTS = frozenset({"elle", "elles", "il", "ils", "on"})
_NEGATION_CLITICS = frozenset({"en", "l'", "la", "le", "les", "leur", "lui", "s'", "se", "y"})
_POSTVERBAL_ADVERBS = frozenset(
    {
        "ainsi",
        "alors",
        "aussi",
        "bien",
        "cependant",
        "deja",
        "donc",
        "encore",
        "enfin",
        "ensuite",
        "guere",
        "jamais",
        "mal",
        "meme",
        "neanmoins",
        "non",
        "pas",
        "plus",
        "point",
        "toujours",
        "toutefois",
    }
)
_OBLIGATION_PARTICIPLES = frozenset(
    {
        "appele",
        "appelee",
        "appelees",
        "appeles",
        "autorise",
        "autorisee",
        "autorisees",
        "autorises",
        "charge",
        "chargee",
        "chargees",
        "charges",
        "contraint",
        "contrainte",
        "contraintes",
        "contraints",
        "fonde",
        "fondee",
        "fondees",
        "fondes",
        "habilite",
        "habilitee",
        "habilitees",
        "habilites",
        "oblige",
        "obligee",
        "obligees",
        "obliges",
        "tenu",
        "tenue",
        "tenues",
        "tenus",
    }
)
_INTRANSITIVE_ETRE_PARTICIPLES = frozenset(
    {
        "alle",
        "allee",
        "allees",
        "alles",
        "apparu",
        "apparue",
        "apparues",
        "apparus",
        "arrive",
        "arrivee",
        "arrivees",
        "arrives",
        "decede",
        "decedee",
        "decedees",
        "decedes",
        "devenu",
        "devenue",
        "devenues",
        "devenus",
        "echu",
        "echue",
        "echues",
        "echus",
        "entre",
        "entree",
        "entrees",
        "entres",
        "expire",
        "expiree",
        "expirees",
        "expires",
        "intervenu",
        "intervenue",
        "intervenues",
        "intervenus",
        "mort",
        "morte",
        "mortes",
        "morts",
        "ne",
        "nee",
        "nees",
        "nes",
        "parti",
        "partie",
        "parties",
        "partis",
        "reste",
        "restee",
        "restees",
        "restes",
        "sorti",
        "sortie",
        "sorties",
        "sortis",
        "survenu",
        "survenue",
        "survenues",
        "survenus",
        "tombe",
        "tombee",
        "tombees",
        "tombes",
        "venu",
        "venue",
        "venues",
        "venus",
    }
)
_SYMMETRIC_PREDICATES = frozenset(
    {
        "commun",
        "commune",
        "communes",
        "communs",
        "conjoint",
        "conjointe",
        "conjointes",
        "conjoints",
        "egal",
        "egale",
        "egales",
        "egaux",
        "equivalent",
        "equivalente",
        "equivalentes",
        "equivalents",
        "identique",
        "identiques",
        "mutuel",
        "mutuelle",
        "mutuelles",
        "mutuels",
        "reciproque",
        "reciproques",
        "semblable",
        "semblables",
        "similaire",
        "similaires",
        "solidaire",
        "solidaires",
    }
)
_ACTOR_PREDICATES = frozenset(
    {
        "redevable",
        "redevables",
        "responsable",
        "responsables",
        "soumis",
        "soumise",
        "soumises",
        "subordonne",
        "subordonnee",
        "subordonnees",
        "subordonnes",
    }
)
_PATIENT_PREDICATES = frozenset(
    {
        "applicable",
        "applicables",
        "du",
        "due",
        "dues",
        "dus",
        "imputable",
        "imputables",
        "opposable",
        "opposables",
    }
)
_RECEIVE_VERB_STEMS = frozenset(
    {
        "achet",
        "acquer",
        "benefici",
        "demand",
        "dispos",
        "emprunt",
        "encaiss",
        "exig",
        "obten",
        "percev",
        "recev",
        "reclam",
        "recueill",
        "recuper",
        "touch",
    }
)
_CONTROL_VERB_STEMS = frozenset(
    {
        "autoris",
        "condamn",
        "contraint",
        "demand",
        "enjoign",
        "impos",
        "interd",
        "oblig",
        "ordonn",
        "permet",
    }
)
_IRREGULAR_VERB_STEMS: dict[str, str] = {
    "acquerir": "acquer",
    "acquiert": "acquer",
    "acquierent": "acquer",
    "acquis": "acquer",
    "acquise": "acquer",
    "acquises": "acquer",
    "appartiennent": "apparten",
    "appartient": "apparten",
    "appartenir": "apparten",
    "conclu": "concl",
    "conclue": "concl",
    "conclues": "concl",
    "concluent": "concl",
    "conclure": "concl",
    "conclus": "concl",
    "conclut": "concl",
    "contraint": "contraint",
    "contrainte": "contraint",
    "devra": "devoir",
    "devront": "devoir",
    "doit": "devoir",
    "doivent": "devoir",
    "du": "devoir",
    "due": "devoir",
    "dues": "devoir",
    "dus": "devoir",
    "enjoint": "enjoign",
    "enjoignent": "enjoign",
    "envoie": "envoi",
    "envoient": "envoi",
    "envoye": "envoi",
    "envoyee": "envoi",
    "envoyer": "envoi",
    "fait": "fair",
    "faite": "fair",
    "faites": "fair",
    "faits": "fair",
    "font": "fair",
    "interdit": "interd",
    "interdite": "interd",
    "interdire": "interd",
    "met": "mett",
    "mettent": "mett",
    "mettre": "mett",
    "mis": "mett",
    "mise": "mett",
    "mises": "mett",
    "obtenir": "obten",
    "obtenu": "obten",
    "obtenue": "obten",
    "obtenues": "obten",
    "obtenus": "obten",
    "obtiennent": "obten",
    "obtient": "obten",
    "paie": "pai",
    "paient": "pai",
    "paiera": "pai",
    "paieront": "pai",
    "paye": "pai",
    "payee": "pai",
    "payees": "pai",
    "payent": "pai",
    "payer": "pai",
    "payes": "pai",
    "percevoir": "percev",
    "percoit": "percev",
    "percoivent": "percev",
    "percu": "percev",
    "percue": "percev",
    "percues": "percev",
    "percus": "percev",
    "permet": "permet",
    "permettent": "permet",
    "permettre": "permet",
    "permis": "permet",
    "permise": "permet",
    "prend": "prend",
    "prendre": "prend",
    "prennent": "prend",
    "pris": "prend",
    "prise": "prend",
    "prises": "prend",
    "prevenir": "preven",
    "prevenu": "preven",
    "prevenue": "preven",
    "previennent": "preven",
    "previent": "preven",
    "recevoir": "recev",
    "recoit": "recev",
    "recoivent": "recev",
    "recu": "recev",
    "recue": "recev",
    "recues": "recev",
    "recus": "recev",
    "remet": "remett",
    "remettent": "remett",
    "remettre": "remett",
    "remis": "remett",
    "remise": "remett",
    "remises": "remett",
    "rend": "rend",
    "rendent": "rend",
    "rendre": "rend",
    "rendu": "rend",
    "rendue": "rend",
    "renvoie": "renvoi",
    "renvoient": "renvoi",
    "renvoye": "renvoi",
    "renvoyee": "renvoi",
    "renvoyer": "renvoi",
    "romp": "romp",
    "rompent": "romp",
    "rompre": "romp",
    "rompt": "romp",
    "rompu": "romp",
    "rompue": "romp",
    "transmet": "transmett",
    "transmettent": "transmett",
    "transmettre": "transmett",
    "transmis": "transmett",
    "transmise": "transmett",
    "transmises": "transmett",
    "vend": "vend",
    "vendent": "vend",
    "vendre": "vend",
    "vendu": "vend",
    "vendue": "vend",
    "vendues": "vend",
    "vendus": "vend",
}
_MULTIWORD_PREPOSITIONS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("a", "la", "charge", "de"), "actor"),
    (("a", "la", "charge", "du"), "actor"),
    (("a", "la", "charge", "des"), "actor"),
    (("a", "la", "charge", "d'"), "actor"),
    (("aux", "frais", "de"), "actor"),
    (("aux", "frais", "du"), "actor"),
    (("aux", "frais", "des"), "actor"),
    (("aux", "frais", "d'"), "actor"),
    (("aux", "depens", "de"), "actor"),
    (("aux", "depens", "du"), "actor"),
    (("aux", "depens", "des"), "actor"),
    (("aux", "depens", "d'"), "actor"),
    (("de", "la", "part", "de"), "actor"),
    (("de", "la", "part", "du"), "actor"),
    (("de", "la", "part", "des"), "actor"),
    (("de", "la", "part", "d'"), "actor"),
    (("au", "profit", "de"), "recipient"),
    (("au", "profit", "du"), "recipient"),
    (("au", "profit", "des"), "recipient"),
    (("au", "profit", "d'"), "recipient"),
    (("en", "faveur", "de"), "recipient"),
    (("en", "faveur", "du"), "recipient"),
    (("en", "faveur", "des"), "recipient"),
    (("en", "faveur", "d'"), "recipient"),
    (("au", "benefice", "de"), "recipient"),
    (("au", "benefice", "du"), "recipient"),
    (("au", "benefice", "des"), "recipient"),
    (("au", "benefice", "d'"), "recipient"),
    (("a", "l'", "egard", "de"), "recipient"),
    (("a", "l'", "egard", "du"), "recipient"),
    (("a", "l'", "egard", "des"), "recipient"),
    (("a", "l'", "egard", "d'"), "recipient"),
    (("a", "l'", "encontre", "de"), "recipient"),
    (("a", "l'", "encontre", "du"), "recipient"),
    (("a", "l'", "encontre", "des"), "recipient"),
    (("a", "l'", "encontre", "d'"), "recipient"),
    (("a", "destination", "de"), "recipient"),
    (("a", "destination", "du"), "recipient"),
    (("a", "destination", "des"), "recipient"),
    (("a", "destination", "d'"), "recipient"),
    (("a", "l'", "attention", "de"), "recipient"),
    (("a", "l'", "attention", "du"), "recipient"),
    (("a", "l'", "attention", "des"), "recipient"),
    (("a", "l'", "attention", "d'"), "recipient"),
    (("a", "l'", "intention", "de"), "recipient"),
    (("a", "l'", "intention", "du"), "recipient"),
    (("a", "l'", "intention", "des"), "recipient"),
    (("a", "l'", "intention", "d'"), "recipient"),
    (("au", "prejudice", "de"), "recipient"),
    (("au", "prejudice", "du"), "recipient"),
    (("au", "prejudice", "des"), "recipient"),
    (("au", "prejudice", "d'"), "recipient"),
    (("au", "detriment", "de"), "recipient"),
    (("au", "detriment", "du"), "recipient"),
    (("au", "detriment", "des"), "recipient"),
    (("au", "detriment", "d'"), "recipient"),
    (("vis-a-vis", "de"), "recipient"),
    (("vis-a-vis", "du"), "recipient"),
    (("vis-a-vis", "des"), "recipient"),
    (("vis-a-vis", "d'"), "recipient"),
    (("pour", "le", "compte", "de"), "recipient"),
    (("pour", "le", "compte", "du"), "recipient"),
    (("pour", "le", "compte", "des"), "recipient"),
    (("pour", "le", "compte", "d'"), "recipient"),
    (("aupres", "de"), "aupres"),
    (("aupres", "du"), "aupres"),
    (("aupres", "des"), "aupres"),
    (("aupres", "d'"), "aupres"),
    (("a", "compter", "de"), "adverbial"),
    (("a", "compter", "du"), "adverbial"),
    (("a", "partir", "de"), "adverbial"),
    (("a", "partir", "du"), "adverbial"),
    (("a", "defaut", "de"), "adverbial"),
    (("a", "defaut", "d'"), "adverbial"),
    (("a", "titre", "de"), "adverbial"),
    (("a", "titre", "d'"), "adverbial"),
    (("a", "raison", "de"), "adverbial"),
    (("a", "hauteur", "de"), "adverbial"),
    (("a", "concurrence", "de"), "adverbial"),
    (("a", "l'", "issue", "de"), "adverbial"),
    (("a", "l'", "issue", "du"), "adverbial"),
    (("a", "l'", "expiration", "de"), "adverbial"),
    (("a", "l'", "expiration", "du"), "adverbial"),
    (("a", "la", "fin", "de"), "adverbial"),
    (("a", "la", "fin", "du"), "adverbial"),
    (("a", "la", "date", "de"), "adverbial"),
    (("a", "la", "date", "du"), "adverbial"),
    (("a", "l'", "occasion", "de"), "adverbial"),
    (("a", "l'", "occasion", "du"), "adverbial"),
    (("a", "peine", "de"), "adverbial"),
    (("a", "moins", "de"), "adverbial"),
    (("a", "proportion", "de"), "adverbial"),
    (("a", "l'", "exception", "de"), "adverbial"),
    (("a", "l'", "exception", "du"), "adverbial"),
    (("a", "la", "suite", "de"), "adverbial"),
    (("a", "la", "suite", "du"), "adverbial"),
    (("par", "rapport", "a"), "adverbial"),
    (("par", "rapport", "au"), "adverbial"),
    (("par", "rapport", "aux"), "adverbial"),
    (("conformement", "a"), "adverbial"),
    (("conformement", "au"), "adverbial"),
    (("conformement", "aux"), "adverbial"),
    (("relativement", "a"), "adverbial"),
    (("relativement", "au"), "adverbial"),
    (("relativement", "aux"), "adverbial"),
    (("jusqu'", "a"), "adverbial"),
    (("jusqu'", "au"), "adverbial"),
    (("jusqu'", "aux"), "adverbial"),
    (("quant", "a"), "adverbial"),
    (("quant", "au"), "adverbial"),
    (("quant", "aux"), "adverbial"),
    (("suite", "a"), "adverbial"),
    (("suite", "au"), "adverbial"),
    (("suite", "aux"), "adverbial"),
    (("grace", "a"), "adverbial"),
    (("grace", "au"), "adverbial"),
    (("grace", "aux"), "adverbial"),
    (("face", "a"), "adverbial"),
    (("face", "au"), "adverbial"),
    (("face", "aux"), "adverbial"),
    (("en", "cas", "de"), "adverbial"),
    (("en", "cas", "d'"), "adverbial"),
)
_BARE_PAR_INSTRUMENTS = frozenset(
    {
        "acte",
        "an",
        "annee",
        "courrier",
        "ecrit",
        "exploit",
        "huissier",
        "jour",
        "lettre",
        "mail",
        "mois",
        "semaine",
        "voie",
    }
)
_ENTITY_STOPWORDS = frozenset(
    _STOPWORDS
    | _SHORT_STOPWORDS
    | _DETERMINERS
    | _PREPOSITION_STARTERS
    | _CLAUSE_LEAD_CONJUNCTIONS
    | _POSTVERBAL_ADVERBS
    | {"aucun", "aucune", "chacun", "chacune", "nul", "nulle", "personne", "rien"}
)
_ADVERBIAL_PREPOSITIONS = frozenset(
    {
        "apres",
        "avant",
        "chez",
        "concernant",
        "conformement",
        "dans",
        "depuis",
        "durant",
        "en",
        "hormis",
        "jusqu'",
        "jusque",
        "lors",
        "malgre",
        "moyennant",
        "outre",
        "pendant",
        "pour",
        "sans",
        "sauf",
        "selon",
        "sous",
        "suivant",
        "sur",
    }
)
_RECIPIENT_PREPOSITIONS = frozenset({"a", "au", "aux", "contre", "envers", "vers"})
_SYMMETRIC_PREPOSITIONS = frozenset({"avec", "entre"})
_PROPER_HEAD_NOUNS = frozenset(
    {
        "article",
        "avenue",
        "cabinet",
        "chambre",
        "code",
        "commission",
        "cour",
        "direction",
        "entreprise",
        "etablissement",
        "fonds",
        "groupe",
        "institut",
        "journal",
        "loi",
        "madame",
        "maitre",
        "ministere",
        "monsieur",
        "office",
        "ordonnance",
        "partie",
        "rue",
        "service",
        "societe",
        "tribunal",
    }
)


@dataclass(frozen=True)
class _ClauseToken:
    raw: str
    lower: str


@dataclass(frozen=True)
class _ClauseRoles:
    actors: frozenset[str]
    patients: frozenset[str]
    recipients: frozenset[str]
    verb_stem: str
    all_tokens: frozenset[str]

    @property
    def targets(self) -> frozenset[str]:
        return self.patients | self.recipients

    @property
    def actor_only(self) -> frozenset[str]:
        return self.actors - self.targets

    @property
    def target_only(self) -> frozenset[str]:
        return self.targets - self.actors

    @property
    def recipient_only(self) -> frozenset[str]:
        return self.recipients - self.actors

    @property
    def patient_only(self) -> frozenset[str]:
        return self.patients - self.actors

    def is_empty(self) -> bool:
        return not (self.actors or self.patients or self.recipients)


@dataclass(frozen=True)
class GroundingClaim:
    """One answer segment and its deterministic evidence verdict."""

    index: int
    text: str
    span_start: int
    span_end: int
    source_numbers: tuple[int, ...]
    valid_source_numbers: tuple[int, ...]
    support_score: float
    status: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "text": self.text,
            "span": {"start": self.span_start, "end": self.span_end},
            "source_numbers": list(self.source_numbers),
            "valid_source_numbers": list(self.valid_source_numbers),
            "support_score": self.support_score,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GroundingReport:
    """Aggregate claim-level grounding metrics for one answer."""

    claims: tuple[GroundingClaim, ...]
    source_count: int
    minimum_support: float

    @property
    def supported_count(self) -> int:
        return sum(claim.status == "supported" for claim in self.claims)

    @property
    def unsupported_count(self) -> int:
        return sum(claim.status == "unsupported" for claim in self.claims)

    @property
    def abstained_count(self) -> int:
        return sum(claim.status == "abstained" for claim in self.claims)

    @property
    def unsupported_claim_rate(self) -> float:
        denominator = self.supported_count + self.unsupported_count
        if denominator == 0:
            return 0.0
        return round(self.unsupported_count / denominator, 4)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "source_count": self.source_count,
            "minimum_support": self.minimum_support,
            "claim_count": len(self.claims),
            "supported_count": self.supported_count,
            "unsupported_count": self.unsupported_count,
            "abstained_count": self.abstained_count,
            "unsupported_claim_rate": self.unsupported_claim_rate,
            "claims": [claim.to_dict() for claim in self.claims],
        }


def _is_non_boundary_period(answer: str, span_start: int, cursor: int) -> bool:
    if cursor + 1 < len(answer) and answer[cursor + 1] == ".":
        return True
    if (
        cursor > 0
        and cursor + 1 < len(answer)
        and answer[cursor - 1].isdigit()
        and answer[cursor + 1].isdigit()
    ):
        return True
    if (
        cursor > 0
        and cursor + 1 < len(answer)
        and answer[cursor - 1].isalpha()
        and answer[cursor + 1].isalpha()
        and (cursor == 1 or not answer[cursor - 2].isalpha())
    ):
        return True

    rest = answer[cursor + 1 :]
    if not rest or rest[0] == "\n" or _TRAILING_FIRST_CITATION.match(rest):
        return False

    prefix = answer[span_start : cursor + 1]
    if _DOTTED_ACRONYM_END_PATTERN.search(prefix):
        if rest[0] in ",;:])":
            return True
        if _LOWERCASE_START.match(rest) or _UPPERCASE_CONTINUATION.match(rest):
            return True
        return bool(re.match(r"[ \t]+\d", rest))

    abbrev_match = _ABBREV_BEFORE_DOT_PATTERN.search(answer, span_start, cursor)
    if abbrev_match and abbrev_match.group(1).casefold() in _LEGAL_ABBREVIATIONS:
        word = abbrev_match.group(1).casefold()
        if word == "etc":
            return rest[0] in ",;:])" or bool(_LOWERCASE_START.match(rest))
        return True

    if rest[0] in ",;:])":
        return True
    if _SPAN_HEADING_PATTERN.match(prefix):
        return True
    return bool(_ARTICLE_REF_BEFORE_DOT_PATTERN.search(prefix) and _LOWERCASE_START.match(rest))


def _consume_trailing_citations(answer: str, span_start: int, end: int) -> int:
    has_citation_in_span = bool(_SOURCE_PATTERN.search(answer, span_start, end))
    first_match = _TRAILING_FIRST_CITATION.match(answer, end)
    if first_match is None:
        return end
    candidate_end = first_match.end()
    while True:
        next_match = _TRAILING_NEXT_CITATION.match(answer, candidate_end)
        if next_match is None:
            break
        candidate_end = next_match.end()
    if _LOWERCASE_START.match(answer, candidate_end):
        return end
    if has_citation_in_span:
        rest_of_line = answer[candidate_end:].split("\n", 1)[0].strip()
        if rest_of_line and rest_of_line != ".":
            return end
    if candidate_end < len(answer) and answer[candidate_end] == ".":
        next_after_dot = candidate_end + 1
        if next_after_dot == len(answer) or answer[next_after_dot].isspace():
            candidate_end = next_after_dot
    return candidate_end


def _claim_spans(answer: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    cursor = 0
    while cursor < len(answer):
        if answer[cursor] == "\n":
            _append_span(spans, answer, start, cursor)
            start = cursor + 1
        elif answer[cursor] in ".!?" and not (
            answer[cursor] == "." and _is_non_boundary_period(answer, start, cursor)
        ):
            end = _consume_trailing_citations(answer, start, cursor + 1)
            _append_span(spans, answer, start, end)
            start = end
            cursor = end - 1
        cursor += 1
    _append_span(spans, answer, start, len(answer))
    return spans


def _append_span(
    spans: list[tuple[int, int]],
    answer: str,
    start: int,
    end: int,
) -> None:
    while start < end and answer[start].isspace():
        start += 1
    while end > start and answer[end - 1].isspace():
        end -= 1
    if start < end:
        spans.append((start, end))


def _fold_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.replace("’", "'"))
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _is_all_caps_segment(text: str) -> bool:
    words = _ALPHA_WORD_PATTERN.findall(text)
    if len(words) < 4:
        return False
    uppercase_words = sum(word.isupper() for word in words)
    return uppercase_words / len(words) >= 0.8


def _extract_acronyms(text: str) -> set[str]:
    without_citations = _SOURCE_PATTERN.sub(" ", text)
    folded = _fold_text(without_citations)
    acronyms: set[str] = set()
    for match in _DOTTED_ACRONYM_PATTERN.findall(folded):
        compact = match.replace(".", "").lower()
        if len(compact) >= 2:
            acronyms.add(compact)
    all_caps = _is_all_caps_segment(folded)
    for token in _UPPERCASE_ACRONYM_PATTERN.findall(folded):
        lowered = token.lower()
        if all_caps:
            if lowered in _LEGAL_ACRONYM_EXPANSIONS and lowered not in _AMBIGUOUS_SHORT_ACRONYMS:
                acronyms.add(lowered)
            continue
        if token in _UPPERCASE_NON_ACRONYMS:
            continue
        acronyms.add(lowered)
    return acronyms


def _acronyms_supported(claim: str, sentence: str) -> bool:
    claim_acronyms = _extract_acronyms(claim)
    if not claim_acronyms:
        return True
    sentence_acronyms = _extract_acronyms(sentence)
    sentence_folded = _fold_text(_SOURCE_PATTERN.sub(" ", sentence)).lower()
    sentence_words = set(re.findall(r"[a-z]+", sentence_folded))
    for acronym in claim_acronyms:
        if acronym in sentence_acronyms:
            continue
        if (
            len(acronym) >= 3
            and acronym not in _AMBIGUOUS_SHORT_ACRONYMS
            and acronym in sentence_words
        ):
            continue
        expansions = _LEGAL_ACRONYM_EXPANSIONS.get(acronym, ())
        if any(re.search(rf"\b{re.escape(expansion)}\b", sentence_folded) for expansion in expansions):
            continue
        return False
    return True


def _content_tokens(text: str) -> set[str]:
    without_citations = _SOURCE_PATTERN.sub(" ", text)
    folded = _fold_text(without_citations)
    folded_lower = folded.lower()
    tokens = {
        token
        for token in _TOKEN_PATTERN.findall(folded_lower)
        if token not in _STOPWORDS
    }
    tokens.update(_extract_acronyms(without_citations))
    for acronym, expansions in _LEGAL_ACRONYM_EXPANSIONS.items():
        if any(re.search(rf"\b{re.escape(expansion)}\b", folded_lower) for expansion in expansions):
            tokens.add(acronym)
    return tokens


def _negation_count(text: str) -> int:
    return len(_NEGATION_PATTERN.findall(text))


def _tokenize_clause(text: str) -> list[_ClauseToken]:
    folded = _fold_text(_SOURCE_PATTERN.sub(" ", text))
    return [
        _ClauseToken(raw=token, lower=token.lower())
        for token in _CLAUSE_TOKEN_PATTERN.findall(folded)
    ]


def _normalize_entity_token(word: str) -> str:
    lower = word.lower()
    if not lower or lower in {",", ";", ":"} or lower.endswith("'"):
        return ""
    if lower in _ENTITY_STOPWORDS:
        return ""
    if lower[0].isdigit():
        return lower
    if len(lower) < 2:
        return ""
    if len(lower) > 4 and lower.endswith("aux"):
        return f"{lower[:-3]}al"
    if len(lower) > 4 and lower.endswith(("eaux", "eux")):
        return lower[:-1]
    if len(lower) > 4 and lower.endswith("s") and not lower.endswith(("is", "us", "ss", "as", "os")):
        return lower[:-1]
    return lower


def _entity_tokens(tokens: list[_ClauseToken]) -> frozenset[str]:
    normalized = (_normalize_entity_token(token.lower) for token in tokens)
    return frozenset(token for token in normalized if token)


def _stem_verb(word: str) -> str:
    lower = word.lower()
    if lower in _IRREGULAR_VERB_STEMS:
        return _IRREGULAR_VERB_STEMS[lower]
    for suffix in (
        "eraient",
        "iraient",
        "erions",
        "irions",
        "eriez",
        "iriez",
        "erent",
        "eront",
        "iront",
        "erait",
        "irait",
        "erais",
        "irais",
        "issent",
        "issait",
        "aient",
        "erons",
        "irons",
        "erez",
        "irez",
        "eons",
        "era",
        "ira",
        "ent",
        "ant",
        "ees",
        "ies",
        "er",
        "ir",
        "re",
        "ee",
        "ie",
        "es",
        "is",
        "e",
        "i",
        "t",
        "s",
        "d",
    ):
        if lower.endswith(suffix) and len(lower) - len(suffix) >= 4:
            return lower[: -len(suffix)]
    for suffix in ("er", "ir", "re", "ee", "es", "e", "i", "t", "s"):
        if lower.endswith(suffix) and len(lower) - len(suffix) >= 3:
            return lower[: -len(suffix)]
    return lower


def _is_adverb(word: str) -> bool:
    return word in _POSTVERBAL_ADVERBS or (len(word) > 5 and word.endswith("ment"))


def _is_infinitive(word: str) -> bool:
    if len(word) < 4 or word in _ENTITY_STOPWORDS:
        return False
    return word.endswith(("er", "ir", "re", "oir"))


def _is_past_participle(word: str) -> bool:
    if len(word) < 2 or word in _ENTITY_STOPWORDS or _is_adverb(word):
        return False
    if word in _INTRANSITIVE_ETRE_PARTICIPLES or word in _OBLIGATION_PARTICIPLES:
        return False
    if word in _SYMMETRIC_PREDICATES or word in _ACTOR_PREDICATES:
        return False
    if word in _PATIENT_PREDICATES:
        return True
    if word.endswith(
        (
            "able",
            "ables",
            "aire",
            "aires",
            "elle",
            "elles",
            "euse",
            "euses",
            "eux",
            "ible",
            "ibles",
            "ique",
            "iques",
            "ive",
            "ives",
            "oire",
            "oires",
        )
    ):
        return False
    return word.endswith(
        (
            "e",
            "ee",
            "es",
            "ees",
            "i",
            "ie",
            "is",
            "ies",
            "u",
            "ue",
            "us",
            "ues",
            "it",
            "ite",
            "its",
            "ites",
            "rt",
            "rte",
            "rts",
            "rtes",
            "nt",
            "nte",
            "nts",
            "ntes",
            "se",
            "ses",
        )
    )


def _match_multiword_prep(tokens: list[_ClauseToken], idx: int) -> tuple[str, int] | None:
    remaining = len(tokens) - idx
    for phrase, kind in _MULTIWORD_PREPOSITIONS:
        length = len(phrase)
        if remaining >= length and all(tokens[idx + offset].lower == phrase[offset] for offset in range(length)):
            return kind, length
    return None


def _skip_adverbs(tokens: list[_ClauseToken], idx: int) -> int:
    while idx < len(tokens) and _is_adverb(tokens[idx].lower):
        idx += 1
    return idx


def _find_verb_group(
    tokens: list[_ClauseToken],
) -> tuple[int, int, str, str] | None:
    """Locate the main finite verb group in a clause.

    Returns (verb_start, verb_end, voice, verb_stem) where voice is one of:
    'active', 'passive', 'receive_active', 'receive_passive', 'incomber',
    'copula', 'symmetric'.
    """
    n = len(tokens)
    if n < 2:
        return None

    # Pass 1: auxiliaries, modals, negation/reflexive clitics
    for i in range(1, n):
        tok = tokens[i].lower
        if tok in {",", ";", ":"}:
            continue
        verb_start = i
        is_reflexive = False
        cur = i

        has_negation = False
        if tok in {"ne", "n'"}:
            has_negation = True
            cur += 1
            if cur >= n:
                continue
        allowed_clitics = (
            _NEGATION_CLITICS
            if (has_negation or (i == 1 and tokens[0].lower in _PRONOUN_SUBJECTS))
            else _PREVERBAL_CLITICS
        )
        while cur < n and tokens[cur].lower in allowed_clitics:
            if tokens[cur].lower in {"se", "s'"}:
                is_reflexive = True
            cur += 1
        if cur >= n:
            continue

        head = tokens[cur].lower
        if head in _ETRE_FORMS:
            nxt = _skip_adverbs(tokens, cur + 1)
            if nxt < n:
                word = tokens[nxt].lower
                if word in _SYMMETRIC_PREDICATES:
                    return verb_start, nxt + 1, "symmetric", _stem_verb(word)
                if word in _ACTOR_PREDICATES:
                    return verb_start, nxt + 1, "active", _stem_verb(word)
                if word in _PATIENT_PREDICATES:
                    return verb_start, nxt + 1, "passive", _stem_verb(word)
                if word in _OBLIGATION_PARTICIPLES:
                    after = _skip_adverbs(tokens, nxt + 1)
                    if after + 1 < n and tokens[after].lower in {"de", "d'", "a"}:
                        inf_idx = _skip_adverbs(tokens, after + 1)
                        if inf_idx < n:
                            inf_word = tokens[inf_idx].lower
                            stem = _stem_verb(inf_word)
                            voice = "receive_active" if stem in _RECEIVE_VERB_STEMS else "active"
                            return verb_start, inf_idx + 1, voice, stem
                if not is_reflexive and _is_past_participle(word):
                    stem = _stem_verb(word)
                    voice = "receive_passive" if stem in _RECEIVE_VERB_STEMS else "passive"
                    return verb_start, nxt + 1, voice, stem
            return verb_start, cur + 1, "copula", "etre"

        if head in _AVOIR_FORMS:
            nxt = _skip_adverbs(tokens, cur + 1)
            if nxt < n:
                word = tokens[nxt].lower
                if word == "ete":
                    part_idx = _skip_adverbs(tokens, nxt + 1)
                    if part_idx < n and _is_past_participle(tokens[part_idx].lower):
                        stem = _stem_verb(tokens[part_idx].lower)
                        voice = "receive_passive" if stem in _RECEIVE_VERB_STEMS else "passive"
                        return verb_start, part_idx + 1, voice, stem
                if word == "droit" and nxt + 1 < n and tokens[nxt + 1].lower in {"a", "au", "aux"}:
                    return verb_start, nxt + 1, "receive_active", "avoir_droit"
                if word in {"le", "l'"} and nxt + 2 < n and tokens[nxt + 1].lower in {"droit", "obligation", "faculte"}:
                    prep_idx = nxt + 2
                    if tokens[prep_idx].lower in {"de", "d'"} and prep_idx + 1 < n:
                        inf_idx = _skip_adverbs(tokens, prep_idx + 1)
                        if inf_idx < n:
                            stem = _stem_verb(tokens[inf_idx].lower)
                            voice = "receive_active" if stem in _RECEIVE_VERB_STEMS else "active"
                            return verb_start, inf_idx + 1, voice, stem
                if _is_past_participle(word) and word not in _DETERMINERS and word not in _PREPOSITION_STARTERS:
                    stem = _stem_verb(word)
                    voice = "receive_active" if stem in _RECEIVE_VERB_STEMS else "active"
                    return verb_start, nxt + 1, voice, stem
            return verb_start, cur + 1, "active", "avoir"

        if head in _MODAL_FORMS:
            nxt = _skip_adverbs(tokens, cur + 1)
            if nxt < n:
                word = tokens[nxt].lower
                if word == "etre":
                    part_idx = _skip_adverbs(tokens, nxt + 1)
                    if part_idx < n and _is_past_participle(tokens[part_idx].lower):
                        stem = _stem_verb(tokens[part_idx].lower)
                        voice = "receive_passive" if stem in _RECEIVE_VERB_STEMS else "passive"
                        return verb_start, part_idx + 1, voice, stem
                if word in {"se", "s'"} and nxt + 1 < n and tokens[nxt + 1].lower == "voir":
                    sub_idx = _skip_adverbs(tokens, nxt + 2)
                    if sub_idx < n:
                        return verb_start, sub_idx + 1, "receive_active", _stem_verb(tokens[sub_idx].lower)
                if _is_infinitive(word):
                    stem = _stem_verb(word)
                    voice = "receive_active" if stem in _RECEIVE_VERB_STEMS else "active"
                    return verb_start, nxt + 1, voice, stem
            return verb_start, cur + 1, "active", _stem_verb(head)

        if is_reflexive and head in {"voit", "voient", "verra", "verront"}:
            nxt = _skip_adverbs(tokens, cur + 1)
            if nxt < n:
                return verb_start, nxt + 1, "receive_active", _stem_verb(tokens[nxt].lower)

        if verb_start < cur:
            # Preceded by ne/n'/se/s'/clitic -> definitely a finite verb
            stem = _stem_verb(head)
            if stem == "incomb":
                return verb_start, cur + 1, "incomber", stem
            voice = "receive_active" if stem in _RECEIVE_VERB_STEMS else "active"
            return verb_start, cur + 1, voice, stem

    # Pass 2: bare finite lexical verb followed by determiner, number, or directional preposition
    def _is_bare_verb_candidate(idx: int) -> bool:
        tok = tokens[idx]
        if tok.lower in _DETERMINERS or tok.lower in _PREPOSITION_STARTERS or tok.lower in _CLAUSE_LEAD_CONJUNCTIONS:
            return False
        if tok.lower in {",", ";", ":"} or tok.lower.endswith("'") or tok.lower[0].isdigit():
            return False
        if _is_adverb(tok.lower):
            return False
        if idx > 0 and tok.raw[0].isupper():
            return False
        return len(tok.lower) >= 2

    for i in range(1, n):
        if not _is_bare_verb_candidate(i):
            continue
        nxt = _skip_adverbs(tokens, i + 1)
        if nxt >= n:
            continue
        nxt_tok = tokens[nxt]
        nxt_low = nxt_tok.lower
        is_strong_trigger = (
            (nxt_low in _DETERMINERS and nxt_low not in {"de", "du", "des", "d'"})
            or nxt_low[0].isdigit()
            or nxt_low in _RECIPIENT_PREPOSITIONS
            or nxt_low in _SYMMETRIC_PREPOSITIONS
            or nxt_low in {"par", "aupres", "vis-a-vis"}
        )
        if is_strong_trigger:
            stem = _stem_verb(tokens[i].lower)
            if stem == "incomb":
                return i, i + 1, "incomber", stem
            voice = "receive_active" if stem in _RECEIVE_VERB_STEMS else "active"
            return i, i + 1, voice, stem

    # Pass 3: bare verb followed by de/du/des/d' (e.g. "dispose de", "beneficie d'un")
    for i in range(1, n):
        if not _is_bare_verb_candidate(i):
            continue
        nxt = _skip_adverbs(tokens, i + 1)
        if nxt >= n:
            continue
        if tokens[nxt].lower in {"de", "du", "des", "d'"}:
            # Skip noun-of-noun inside subject if another verb follows later
            if tokens[i].lower in _PROPER_HEAD_NOUNS:
                continue
            stem = _stem_verb(tokens[i].lower)
            voice = "receive_active" if stem in _RECEIVE_VERB_STEMS else "active"
            return i, i + 1, voice, stem

    # Pass 4: bare verb followed by capitalized proper noun object (e.g. "renvoie Alpha a Beta")
    for i in range(1, n):
        if not _is_bare_verb_candidate(i):
            continue
        if tokens[i].lower in _PROPER_HEAD_NOUNS:
            continue
        nxt = _skip_adverbs(tokens, i + 1)
        if nxt < n and tokens[nxt].raw[0].isupper() and tokens[nxt].lower not in _ENTITY_STOPWORDS:
            stem = _stem_verb(tokens[i].lower)
            voice = "receive_active" if stem in _RECEIVE_VERB_STEMS else "active"
            return i, i + 1, voice, stem

    return None


def _parse_pp_arguments(
    tokens: list[_ClauseToken],
    *,
    allow_bare_object: bool,
    is_passive: bool,
    is_receive: bool,
    is_nominal: bool = False,
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Extract (direct_object, actor_pps, recipient_pps, source_pps) tokens."""
    direct_obj: set[str] = set()
    actor_pps: set[str] = set()
    recipient_pps: set[str] = set()
    source_pps: set[str] = set()

    current_bucket: set[str] | None = direct_obj if allow_bare_object else None
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i].lower
        if tok in {",", ";", ":"}:
            current_bucket = None
            i += 1
            continue

        mw = _match_multiword_prep(tokens, i)
        if mw is not None:
            kind, length = mw
            if kind == "actor":
                current_bucket = actor_pps
            elif kind == "recipient":
                current_bucket = recipient_pps
            elif kind == "aupres":
                current_bucket = source_pps if is_receive else recipient_pps
            else:
                current_bucket = None
            i += length
            continue

        if tok == "par":
            nxt = i + 1
            has_det = nxt < n and (
                tokens[nxt].lower in _DETERMINERS
                or (tokens[nxt].raw[0].isupper() and tokens[nxt].lower not in _BARE_PAR_INSTRUMENTS)
            )
            is_bare_instrument = nxt < n and tokens[nxt].lower in _BARE_PAR_INSTRUMENTS
            if (is_passive or is_nominal) and has_det and not is_bare_instrument:
                current_bucket = actor_pps
            else:
                current_bucket = None
            i += 1
            continue

        if tok in _RECIPIENT_PREPOSITIONS:
            nxt = i + 1
            if tok == "a" and nxt < n and (
                _is_infinitive(tokens[nxt].lower) or tokens[nxt].lower[0].isdigit()
            ):
                current_bucket = None
            else:
                current_bucket = recipient_pps
            i += 1
            continue

        if tok in _SYMMETRIC_PREPOSITIONS or tok in _ADVERBIAL_PREPOSITIONS:
            current_bucket = None
            i += 1
            continue

        if is_receive and tok in {"du", "des"}:
            current_bucket = source_pps
            i += 1
            continue

        if is_receive and tok in {"de", "d'"}:
            nxt = i + 1
            if nxt < n and (
                tokens[nxt].lower in _DETERMINERS
                or tokens[nxt].raw[0].isupper()
            ):
                current_bucket = source_pps
                i += 1
                continue

        if current_bucket is not None:
            norm = _normalize_entity_token(tok)
            if norm:
                current_bucket.add(norm)
        i += 1

    return direct_obj, actor_pps, recipient_pps, source_pps


def _extract_impersonal_obligation(tokens: list[_ClauseToken]) -> _ClauseRoles | None:
    if len(tokens) < 6 or tokens[0].lower != "il":
        return None
    verb = tokens[1].lower
    prep_idx = 2
    if verb in {"appartient", "incombe", "revient"}:
        pass
    elif verb in _ETRE_FORMS and len(tokens) > 4 and tokens[2].lower in {"interdit", "fait"}:
        prep_idx = 3 if tokens[2].lower == "interdit" else 4
    else:
        return None
    if prep_idx >= len(tokens) or tokens[prep_idx].lower not in {"a", "au", "aux"}:
        return None
    de_idx = None
    for idx in range(prep_idx + 1, len(tokens) - 1):
        if tokens[idx].lower in {"de", "d'"} and _is_infinitive(tokens[idx + 1].lower):
            de_idx = idx
            break
    if de_idx is None:
        return None
    actor_tokens = _entity_tokens(tokens[prep_idx + 1 : de_idx])
    inf_word = tokens[de_idx + 1].lower
    stem = _stem_verb(inf_word)
    is_receive = stem in _RECEIVE_VERB_STEMS
    direct_obj, actor_pps, recipient_pps, source_pps = _parse_pp_arguments(
        tokens[de_idx + 2 :],
        allow_bare_object=True,
        is_passive=False,
        is_receive=is_receive,
    )
    if is_receive:
        actors = frozenset(source_pps | actor_pps)
        recipients = frozenset(actor_tokens | recipient_pps)
    else:
        actors = frozenset(actor_tokens | actor_pps)
        recipients = frozenset(recipient_pps)
    patients = frozenset(direct_obj)
    all_tokens = actors | patients | recipients | ({stem} if stem else set())
    return _ClauseRoles(
        actors=actors,
        patients=patients,
        recipients=recipients,
        verb_stem=stem,
        all_tokens=frozenset(all_tokens),
    )


def _extract_embedded_infinitive(
    post_verb: list[_ClauseToken],
    verb_stem: str,
) -> _ClauseRoles | None:
    if verb_stem not in _CONTROL_VERB_STEMS:
        return None
    for idx in range(1, len(post_verb) - 1):
        if post_verb[idx].lower in {"de", "d'", "a"} and _is_infinitive(post_verb[idx + 1].lower):
            controllee = _entity_tokens(post_verb[:idx])
            if not controllee:
                return None
            inf_word = post_verb[idx + 1].lower
            inf_stem = _stem_verb(inf_word)
            is_receive = inf_stem in _RECEIVE_VERB_STEMS
            direct_obj, actor_pps, recipient_pps, source_pps = _parse_pp_arguments(
                post_verb[idx + 2 :],
                allow_bare_object=True,
                is_passive=False,
                is_receive=is_receive,
            )
            if is_receive:
                actors = frozenset(source_pps | actor_pps)
                recipients = frozenset(controllee | recipient_pps)
            else:
                actors = frozenset(controllee | actor_pps)
                recipients = frozenset(recipient_pps)
            patients = frozenset(direct_obj)
            all_tokens = actors | patients | recipients | ({inf_stem} if inf_stem else set())
            return _ClauseRoles(
                actors=actors,
                patients=patients,
                recipients=recipients,
                verb_stem=inf_stem,
                all_tokens=frozenset(all_tokens),
            )
    return None


def _parse_single_clause(tokens: list[_ClauseToken]) -> list[_ClauseRoles]:
    while tokens and tokens[0].lower in _CLAUSE_LEAD_CONJUNCTIONS:
        tokens = tokens[1:]
    if not tokens:
        return []

    impersonal = _extract_impersonal_obligation(tokens)
    if impersonal is not None:
        return [impersonal]

    fronted_actors: set[str] = set()
    fronted_recipients: set[str] = set()
    # Peel leading adverbial or fronted directional PPs before commas
    while True:
        comma_idx = next((idx for idx, tok in enumerate(tokens) if tok.lower == ","), None)
        if comma_idx is None:
            break
        prefix = tokens[:comma_idx]
        if not prefix:
            tokens = tokens[comma_idx + 1 :]
            continue
        if _find_verb_group(prefix) is not None:
            break
        if prefix[0].lower in _PREPOSITION_STARTERS or _match_multiword_prep(prefix, 0) is not None:
            _, f_act, f_rec, _ = _parse_pp_arguments(
                prefix,
                allow_bare_object=False,
                is_passive=True,
                is_receive=False,
                is_nominal=True,
            )
            fronted_actors.update(f_act)
            fronted_recipients.update(f_rec)
            tokens = tokens[comma_idx + 1 :]
            while tokens and tokens[0].lower in _CLAUSE_LEAD_CONJUNCTIONS:
                tokens = tokens[1:]
            continue
        break

    if not tokens:
        return []

    impersonal = _extract_impersonal_obligation(tokens)
    if impersonal is not None:
        return [impersonal]

    verb_info = _find_verb_group(tokens)
    if verb_info is None:
        _, nom_actors, nom_recipients, _ = _parse_pp_arguments(
            tokens,
            allow_bare_object=False,
            is_passive=False,
            is_receive=False,
            is_nominal=True,
        )
        actors = frozenset(fronted_actors | nom_actors)
        recipients = frozenset(fronted_recipients | nom_recipients)
        if actors or recipients:
            return [
                _ClauseRoles(
                    actors=actors,
                    patients=frozenset(),
                    recipients=recipients,
                    verb_stem="",
                    all_tokens=actors | recipients,
                )
            ]
        return []

    verb_start, verb_end, voice, verb_stem = verb_info
    if voice == "symmetric":
        return []

    subject_slice = tokens[:verb_start]
    post_verb = tokens[verb_end:]
    subject_tokens = _entity_tokens(subject_slice)

    is_passive = voice in {"passive", "receive_passive"}
    is_receive = voice in {"receive_active", "receive_passive"}

    direct_obj, actor_pps, recipient_pps, source_pps = _parse_pp_arguments(
        post_verb,
        allow_bare_object=not is_passive and voice != "copula",
        is_passive=is_passive,
        is_receive=is_receive,
    )

    if voice == "copula":
        _, subj_actors, subj_recipients, _ = _parse_pp_arguments(
            subject_slice,
            allow_bare_object=False,
            is_passive=False,
            is_receive=False,
            is_nominal=True,
        )
        actors = frozenset(fronted_actors | actor_pps | subj_actors)
        recipients = frozenset(fronted_recipients | recipient_pps | subj_recipients)
        patients = subject_tokens - actors - recipients
        if not actors and not recipients:
            return []
        all_tokens = actors | patients | recipients
        return [
            _ClauseRoles(
                actors=actors,
                patients=patients,
                recipients=recipients,
                verb_stem="",
                all_tokens=all_tokens,
            )
        ]

    if voice == "incomber":
        actors = frozenset(fronted_actors | recipient_pps | actor_pps)
        patients = subject_tokens
        # Separate explicit beneficiary PPs (au profit de / en faveur de) if present
        _, _, ben_pps, _ = _parse_pp_arguments(
            post_verb,
            allow_bare_object=False,
            is_passive=False,
            is_receive=False,
        )
        # Re-scan post_verb for explicit multiword beneficiary vs bare "a/au/aux" obligor
        obligors: set[str] = set()
        beneficiaries: set[str] = set()
        bucket: set[str] | None = None
        i = 0
        while i < len(post_verb):
            mw = _match_multiword_prep(post_verb, i)
            if mw is not None:
                kind, length = mw
                bucket = beneficiaries if kind == "recipient" else (obligors if kind == "actor" else None)
                i += length
                continue
            low = post_verb[i].lower
            if low in {"a", "au", "aux"}:
                bucket = obligors
            elif low in _ADVERBIAL_PREPOSITIONS or low in _SYMMETRIC_PREPOSITIONS or low in {",", ";", ":"}:
                bucket = None
            elif bucket is not None:
                norm = _normalize_entity_token(low)
                if norm:
                    bucket.add(norm)
            i += 1
        actors = frozenset(fronted_actors | obligors) if obligors else actors
        recipients = frozenset(fronted_recipients | beneficiaries)
        _ = ben_pps
    elif voice == "passive":
        actors = frozenset(fronted_actors | actor_pps)
        patients = subject_tokens
        recipients = frozenset(fronted_recipients | recipient_pps)
    elif voice == "receive_active":
        actors = frozenset(fronted_actors | source_pps | actor_pps)
        patients = frozenset(direct_obj)
        recipients = frozenset(subject_tokens | fronted_recipients | recipient_pps)
    elif voice == "receive_passive":
        actors = frozenset(fronted_actors | source_pps)
        patients = subject_tokens
        recipients = frozenset(fronted_recipients | actor_pps | recipient_pps)
    else:
        actors = frozenset(subject_tokens | fronted_actors | actor_pps)
        patients = frozenset(direct_obj)
        recipients = frozenset(fronted_recipients | recipient_pps)

    results: list[_ClauseRoles] = []
    all_tokens = actors | patients | recipients | ({verb_stem} if verb_stem else set())
    main_roles = _ClauseRoles(
        actors=actors,
        patients=patients,
        recipients=recipients,
        verb_stem=verb_stem,
        all_tokens=frozenset(all_tokens),
    )
    if not main_roles.is_empty():
        results.append(main_roles)

    embedded = _extract_embedded_infinitive(post_verb, verb_stem)
    if embedded is not None and not embedded.is_empty():
        results.append(embedded)

    _, subj_actors, subj_recipients, _ = _parse_pp_arguments(
        subject_slice,
        allow_bare_object=False,
        is_passive=False,
        is_receive=False,
        is_nominal=True,
    )
    if subj_actors and subj_recipients:
        nom_actors = frozenset(subj_actors)
        nom_recipients = frozenset(subj_recipients)
        nom_patients = subject_tokens - nom_actors - nom_recipients
        results.append(
            _ClauseRoles(
                actors=nom_actors,
                patients=nom_patients,
                recipients=nom_recipients,
                verb_stem="",
                all_tokens=nom_actors | nom_patients | nom_recipients,
            )
        )

    return results


def _split_into_clauses(tokens: list[_ClauseToken]) -> list[list[_ClauseToken]]:
    # Split first on semicolons and colons
    segments: list[list[_ClauseToken]] = []
    current: list[_ClauseToken] = []
    for token in tokens:
        if token.lower in {";", ":"}:
            if current:
                segments.append(current)
                current = []
        else:
            current.append(token)
    if current:
        segments.append(current)

    # Split each segment on commas or clause conjunctions only when both halves contain a verb
    clauses: list[list[_ClauseToken]] = []
    queue = list(segments)
    while queue:
        seg = queue.pop(0)
        split_done = False
        for idx, tok in enumerate(seg):
            if idx < 2 or idx >= len(seg) - 2:
                continue
            low = tok.lower
            if low in {",", "mais", "cependant", "toutefois", "et", "tandis", "alors", "lorsque", "quand"}:
                left = seg[:idx]
                right = seg[idx + 1 :]
                if low in {"tandis", "alors"} and right and right[0].lower in {"que", "qu'"}:
                    right = right[1:]
                # Strip leading conjunction on right before checking verb presence
                right_probe = right
                while right_probe and right_probe[0].lower in _CLAUSE_LEAD_CONJUNCTIONS:
                    right_probe = right_probe[1:]
                left_probe = left
                while left_probe and left_probe[0].lower in _CLAUSE_LEAD_CONJUNCTIONS:
                    left_probe = left_probe[1:]
                if _find_verb_group(left_probe) is not None and _find_verb_group(right_probe) is not None:
                    queue.insert(0, right)
                    queue.insert(0, left)
                    split_done = True
                    break
        if not split_done and seg:
            clauses.append(seg)
    return clauses


def _extract_sentence_roles(text: str) -> list[_ClauseRoles]:
    tokens = _tokenize_clause(text)
    if not tokens:
        return []
    roles: list[_ClauseRoles] = []
    for clause in _split_into_clauses(tokens):
        roles.extend(_parse_single_clause(clause))
    return roles


def _clauses_conflict(claim_role: _ClauseRoles, evidence_role: _ClauseRoles) -> bool:
    if claim_role.is_empty() or evidence_role.is_empty():
        return False

    # Condition 1: Bidirectional actor <-> target swap (always a directional reversal)
    actor_to_target = bool(claim_role.target_only & evidence_role.actor_only)
    target_to_actor = bool(claim_role.actor_only & evidence_role.target_only)
    if actor_to_target and target_to_actor:
        return True

    # Condition 2: Same-verb single-side role inversion (catches partial-claim reversals)
    if claim_role.verb_stem and claim_role.verb_stem == evidence_role.verb_stem:
        if claim_role.actor_only & evidence_role.recipient_only:
            return True
        if claim_role.recipient_only & evidence_role.actor_only:
            return True
        if claim_role.patient_only & evidence_role.actor_only:
            return True
        if (claim_role.actor_only & evidence_role.patient_only) and (
            claim_role.patients or claim_role.recipients
        ):
            return True

    return False


def _roles_compatible(claim: str, sentence: str) -> bool:
    claim_roles = _extract_sentence_roles(claim)
    if not claim_roles:
        return True
    evidence_roles = _extract_sentence_roles(sentence)
    if not evidence_roles:
        return True

    for claim_role in claim_roles:
        ranked: list[tuple[tuple[int, int, int], _ClauseRoles]] = []
        for ev_role in evidence_roles:
            overlap = len(claim_role.all_tokens & ev_role.all_tokens)
            if overlap < 2:
                continue
            same_verb = 1 if (claim_role.verb_stem and claim_role.verb_stem == ev_role.verb_stem) else 0
            same_kind = 1 if (bool(claim_role.verb_stem) == bool(ev_role.verb_stem)) else 0
            ranked.append(((same_verb, same_kind, overlap), ev_role))
        if not ranked:
            continue
        best_score = max(score for score, _ in ranked)
        best_matches = [ev_role for score, ev_role in ranked if score == best_score]
        if all(_clauses_conflict(claim_role, ev_role) for ev_role in best_matches):
            return False
    return True


def _is_abstention(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    if normalized in _SYSTEM_ABSTENTION_NOTICES:
        return True
    if _SOURCE_PATTERN.search(text) or _NUMBER_PATTERN.search(text):
        return False
    if re.search(r"\b(?:mais|cependant|pourtant|toutefois|donc|ainsi|car)\b|[;:]", normalized):
        return False
    return any(normalized.startswith(marker) for marker in _ABSTENTION_MARKERS)


def _support_score(claim_text: str, evidence: str) -> float:
    claim_tokens = _content_tokens(claim_text)
    if claim_tokens:
        evidence_tokens = _content_tokens(evidence)
        return round(len(claim_tokens & evidence_tokens) / len(claim_tokens), 4)
    claim_short = {
        token
        for token in _SHORT_TOKEN_PATTERN.findall(_fold_text(_SOURCE_PATTERN.sub(" ", claim_text)).lower())
        if token not in _SHORT_STOPWORDS
    }
    if not claim_short:
        return 0.0
    evidence_short = {
        token
        for token in _SHORT_TOKEN_PATTERN.findall(_fold_text(_SOURCE_PATTERN.sub(" ", evidence)).lower())
        if token not in _SHORT_STOPWORDS
    }
    return round(len(claim_short & evidence_short) / len(claim_short), 4)


def _evidence_verdict(text: str, evidence: str, minimum_support: float) -> tuple[float, str]:
    claim = _SOURCE_PATTERN.sub(" ", text)
    numbers = set(_NUMBER_PATTERN.findall(claim))
    sentences = [evidence[start:end] for start, end in _claim_spans(evidence)]
    candidates = [sentence for sentence in sentences if numbers <= set(_NUMBER_PATTERN.findall(sentence))]
    if not candidates:
        return 0.0, "unsupported_numeric_value"
    candidates = [
        sentence for sentence in candidates
        if _acronyms_supported(claim, sentence)
    ]
    if not candidates:
        return 0.0, "unsupported_acronym"
    claim_negations = _negation_count(claim)
    candidates = [
        sentence for sentence in candidates
        if claim_negations == _negation_count(sentence)
    ]
    if not candidates:
        return 0.0, "conflicting_polarity"
    candidates = [
        sentence for sentence in candidates
        if _roles_compatible(claim, sentence)
    ]
    if not candidates:
        return 0.0, "conflicting_role_direction"
    score = max((_support_score(claim, sentence) for sentence in candidates), default=0.0)
    return score, "evidence_overlap" if score >= minimum_support else "low_evidence_overlap"


def verify_grounding(
    answer: str,
    retrieved_chunks: list[dict],
    minimum_support: float = DEFAULT_MINIMUM_SUPPORT,
) -> GroundingReport:
    """Verify source references and lexical evidence for every answer segment."""
    if not 0 <= minimum_support <= 1:
        raise ValueError("minimum_support must be between 0 and 1")
    normalized = normalize_citations(answer)
    claims: list[GroundingClaim] = []
    source_count = len(retrieved_chunks)
    for index, (start, end) in enumerate(_claim_spans(normalized), 1):
        text = normalized[start:end].strip()
        source_numbers = tuple(sorted({int(number) for number in _SOURCE_PATTERN.findall(text)}))
        valid_source_numbers = tuple(number for number in source_numbers if 1 <= number <= source_count)
        invalid_source_numbers = tuple(number for number in source_numbers if not 1 <= number <= source_count)
        if _is_abstention(text):
            status = "abstained"
            reason = "explicit_source_limitation"
            score = 1.0
        elif not source_numbers:
            status = "unsupported"
            reason = "missing_source_citation"
            score = 0.0
        elif invalid_source_numbers:
            status = "unsupported"
            reason = f"invalid_source_citation:{','.join(map(str, invalid_source_numbers))}"
            score = 0.0
        else:
            evidence = "\n".join(
                retrieved_chunks[number - 1].get("text", "")
                for number in valid_source_numbers
            )
            score, reason = _evidence_verdict(text, evidence, minimum_support)
            status = "supported" if reason == "evidence_overlap" else "unsupported"
        claims.append(
            GroundingClaim(
                index=index,
                text=text,
                span_start=start,
                span_end=end,
                source_numbers=source_numbers,
                valid_source_numbers=valid_source_numbers,
                support_score=score,
                status=status,
                reason=reason,
            )
        )
    return GroundingReport(tuple(claims), source_count, minimum_support)


def _remove_invalid_citations(text: str, source_count: int) -> str:
    def replace(match: re.Match[str]) -> str:
        return match.group(0) if 1 <= int(match.group(1)) <= source_count else ""

    return _SOURCE_PATTERN.sub(replace, text)


def enforce_grounding(
    answer: str,
    retrieved_chunks: list[dict],
    minimum_support: float = DEFAULT_MINIMUM_SUPPORT,
) -> tuple[str, GroundingReport]:
    """Keep supported claims, remove unsupported ones, and append an abstention."""
    normalized = normalize_citations(answer).strip()
    report = verify_grounding(normalized, retrieved_chunks, minimum_support)
    kept_claims = [
        _remove_invalid_citations(claim.text, report.source_count).strip()
        for claim in report.claims
        if claim.status in {"supported", "abstained"}
    ]
    kept_claims = [claim for claim in kept_claims if claim]
    if report.unsupported_count:
        kept_claims.append(_PARTIAL_ABSTENTION_NOTICE)
    if not kept_claims:
        kept_claims = [_EMPTY_ABSTENTION_LEAD, _EMPTY_ABSTENTION_TAIL]
    return "\n".join(kept_claims), report
