import pytest

from tools.review_empty_sources import inspect_page, source_page_url


def _page(content):
    return f'<main><div class="documentContent"><div id="content">{content}</div></div></main>'.encode()


def test_editorial_note_is_evidence_and_never_becomes_an_operative_body():
    evidence = inspect_page(_page('<header class="documentTitle">Loi test</header>'
                                  '<aside class="editoNotes">Texte abrogé selon la note.</aside><nav>Haut</nav>'))
    assert (evidence["candidate_body_chars_after_metadata_exclusion"], evidence["editorial_repeal_mention"],
            evidence["legal_review"]) == (0, True, "pending")


def test_source_repeal_banner_does_not_repair_a_legal_status():
    evidence = inspect_page(_page('<header><span class="abrogatedText">Abrogé</span></header>'))
    assert (evidence["html_repeal_banner"], evidence.get("abrogee")) == (True, None)


def test_real_body_is_not_discarded_with_source_chrome():
    evidence = inspect_page(_page('<header>Titre</header><div class="block">Article 1 : règle.</div><aside>Versions</aside>'))
    assert evidence["classification"] == "body_present_review_extraction"


def test_historical_navigation_is_not_a_current_body():
    evidence = inspect_page(_page('<p class="histoLink">Ancienne rédaction</p><aside class="histoList">Historique</aside>'))
    assert evidence["classification"] == "no_body_after_metadata_exclusion"


def test_unexpected_source_markup_is_reported_as_a_failure():
    with pytest.raises(ValueError, match="container missing"):
        inspect_page(b"<html>Service unavailable</html>")


def test_canonical_page_request_preserves_the_exact_source_version():
    source = "https://legimonaco.mc/tnc/loi/1988/06-27-1.113@2020.12.19"
    assert source_page_url(source) == source + "/"


def test_canonical_page_request_rejects_a_different_source():
    with pytest.raises(ValueError):
        source_page_url("https://other.example/tnc/loi/1@2020.01.01")
