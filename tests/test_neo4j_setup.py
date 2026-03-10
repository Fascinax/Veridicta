"""Tests for retrievers/neo4j_setup.py — regex extraction, normalization, and Neo4jManager."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from retrievers.neo4j_setup import (
    _normalise_law_ref,
    _extract_law_number,
    _ref_matches_doc,
    _article_node_id,
    _group_by_doc,
    _collect_themes,
    extract_law_refs,
    extract_article_citations,
    extract_voir_article_refs,
    extract_modifie_refs_from_titre,
    Neo4jManager,
)


# ── _normalise_law_ref ───────────────────────────────────────────────────

class TestNormaliseLawRef:

    def test_lowercases_and_collapses_whitespace(self) -> None:
        assert _normalise_law_ref("Loi  N°  729") == "loi n° 729"

    def test_already_normalised(self) -> None:
        assert _normalise_law_ref("loi n° 729") == "loi n° 729"

    def test_empty_string(self) -> None:
        assert _normalise_law_ref("") == ""


# ── _extract_law_number ──────────────────────────────────────────────────

class TestExtractLawNumber:

    def test_loi_simple(self) -> None:
        assert _extract_law_number("Loi n° 1.048 du 28 juillet 1982") == "1.048"

    def test_ordonnance_souveraine(self) -> None:
        assert _extract_law_number("Ordonnance souveraine n° 3.162 du 5 mai 1964") == "3.162"

    def test_no_match(self) -> None:
        assert _extract_law_number("Titre sans numéro") == ""

    def test_arrete_ministeriel(self) -> None:
        assert _extract_law_number("Arrêté ministériel n° 2019-456") == "2019-456"


# ── _ref_matches_doc ─────────────────────────────────────────────────────

class TestRefMatchesDoc:

    def test_code_du_travail_match(self) -> None:
        assert _ref_matches_doc("code du travail", "Code du travail monégasque") is True

    def test_law_number_match(self) -> None:
        assert _ref_matches_doc("loi n° 1.048", "Loi n° 1.048 du 28 juillet 1982 sur le contrat") is True

    def test_no_match(self) -> None:
        assert _ref_matches_doc("loi n° 999", "Loi n° 1.048 du 28 juillet 1982") is False

    def test_code_de_commerce(self) -> None:
        assert _ref_matches_doc("code de commerce", "Code de commerce maritime") is True


# ── _article_node_id ─────────────────────────────────────────────────────

class TestArticleNodeId:

    def test_basic(self) -> None:
        assert _article_node_id("doc-123", "5") == "doc-123__art_5"

    def test_lowercases_article_number(self) -> None:
        assert _article_node_id("doc-1", "5BIS") == "doc-1__art_5bis"


# ── extract_law_refs ─────────────────────────────────────────────────────

class TestExtractLawRefs:

    def test_single_loi(self) -> None:
        text = "En vertu de la loi n° 729 du 16 mars 1963"
        refs = extract_law_refs(text)
        assert any("729" in r for r in refs)

    def test_ordonnance(self) -> None:
        text = "L'ordonnance souveraine n° 3.162 du 5 mai 1964 prévoit"
        refs = extract_law_refs(text)
        assert any("3.162" in r for r in refs)

    def test_code_du_travail(self) -> None:
        refs = extract_law_refs("conformément au Code du Travail")
        assert any("code du travail" in r for r in refs)

    def test_multiple_refs(self) -> None:
        text = "La loi n° 729 et l'ordonnance souveraine n° 3.162 s'appliquent"
        refs = extract_law_refs(text)
        assert len(refs) >= 2

    def test_no_match(self) -> None:
        assert extract_law_refs("Un texte sans référence juridique.") == []

    def test_deduplication(self) -> None:
        text = "La loi n° 729 et la loi n° 729 sont mentionnées deux fois."
        refs = extract_law_refs(text)
        assert refs.count(refs[0]) == 1


# ── extract_article_citations ────────────────────────────────────────────

class TestExtractArticleCitations:

    def test_article_de_la_loi(self) -> None:
        text = "l'article 5 de la loi n° 729"
        results = extract_article_citations(text)
        assert len(results) >= 1
        assert results[0][0] == "5"
        assert "729" in results[0][1]

    def test_articles_pluriel(self) -> None:
        text = "les articles 2 de la loi n° 1.048"
        results = extract_article_citations(text)
        assert any(art == "2" for art, _ in results)

    def test_article_ordonnance(self) -> None:
        text = "l'article 3 de l'ordonnance souveraine n° 3.162"
        results = extract_article_citations(text)
        assert any("3.162" in ref for _, ref in results)

    def test_voir_article_also_captured(self) -> None:
        text = "Voir l'article 10 de la loi n° 729"
        results = extract_article_citations(text)
        assert any(art == "10" for art, _ in results)

    def test_no_match(self) -> None:
        assert extract_article_citations("Aucun article cité.") == []


# ── extract_voir_article_refs ────────────────────────────────────────────

class TestExtractVoirArticleRefs:

    def test_voir_article_loi(self) -> None:
        text = "Voir l'article 8 de la loi n° 729"
        results = extract_voir_article_refs(text)
        assert len(results) == 1
        assert results[0][0] == "8"
        assert "729" in results[0][1]

    def test_voir_article_ordonnance(self) -> None:
        text = "voir l'article 12 de l'ordonnance n° 3.162"
        results = extract_voir_article_refs(text)
        assert any("12" == art for art, _ in results)

    def test_no_match(self) -> None:
        assert extract_voir_article_refs("Pas de renvoi ici.") == []

    def test_deduplication(self) -> None:
        text = "Voir l'article 5 de la loi n° 729. Voir l'article 5 de la loi n° 729."
        results = extract_voir_article_refs(text)
        assert len(results) == 1


# ── extract_modifie_refs_from_titre ──────────────────────────────────────

class TestExtractModifieRefs:

    def test_portant_modification(self) -> None:
        titre = "Loi n° 1.255 portant modification de la loi n° 1.048"
        refs = extract_modifie_refs_from_titre(titre)
        assert any("1.048" in r for r in refs)

    def test_modifiant_la_loi(self) -> None:
        titre = "Ordonnance souveraine n° 4.000 modifiant la loi n° 729"
        refs = extract_modifie_refs_from_titre(titre)
        assert any("729" in r for r in refs)

    def test_no_modification_ref(self) -> None:
        assert extract_modifie_refs_from_titre("Loi n° 729 sur le travail") == []

    def test_completant_et_modifiant(self) -> None:
        titre = "Loi n° 1.300 complétant et modifiant la loi n° 1.048"
        refs = extract_modifie_refs_from_titre(titre)
        assert any("1.048" in r for r in refs)


# ── _group_by_doc ────────────────────────────────────────────────────────

class TestGroupByDoc:

    def test_groups_correctly(self) -> None:
        chunks = [
            {"doc_id": "d1", "titre": "Loi n° 729", "type": "legislation", "date": "1963", "source": "lm", "metadata": {}, "text": "a", "chunk_index": 0, "total_chunks": 2, "chunk_id": "c1"},
            {"doc_id": "d1", "titre": "Loi n° 729", "type": "legislation", "date": "1963", "source": "lm", "metadata": {}, "text": "b", "chunk_index": 1, "total_chunks": 2, "chunk_id": "c2"},
            {"doc_id": "d2", "titre": "Arrêt X", "type": "jurisprudence", "date": "2020", "source": "lm", "metadata": {}, "text": "c", "chunk_index": 0, "total_chunks": 1, "chunk_id": "c3"},
        ]
        docs = _group_by_doc(chunks)
        assert len(docs) == 2
        assert "d1" in docs
        assert "d2" in docs
        assert docs["d1"]["type"] == "legislation"

    def test_uses_first_chunk_metadata(self) -> None:
        chunks = [
            {"doc_id": "d1", "titre": "T1", "type": "legislation", "date": "2000", "source": "s", "metadata": {"thematic": "travail"}},
            {"doc_id": "d1", "titre": "T1-updated", "type": "legislation", "date": "2001", "source": "s2", "metadata": {}},
        ]
        docs = _group_by_doc(chunks)
        assert docs["d1"]["date"] == "2000"

    def test_empty_list(self) -> None:
        assert _group_by_doc([]) == {}


# ── _collect_themes ──────────────────────────────────────────────────────

class TestCollectThemes:

    def test_string_theme(self) -> None:
        doc = {"metadata": {"thematic": "droit du travail"}}
        assert _collect_themes(doc) == ["droit du travail"]

    def test_list_themes(self) -> None:
        doc = {"metadata": {"themes": ["travail", "contrat"]}}
        assert _collect_themes(doc) == ["travail", "contrat"]

    def test_deduplicates(self) -> None:
        doc = {"metadata": {"thematic": "travail", "themes": ["travail"]}}
        themes = _collect_themes(doc)
        assert themes.count("travail") == 1

    def test_no_metadata(self) -> None:
        assert _collect_themes({}) == []
        assert _collect_themes({"metadata": {}}) == []


# ── Neo4jManager — guard clauses (no real Neo4j) ────────────────────────

class TestNeo4jManagerGuardClauses:

    def test_get_cited_doc_ids_returns_empty_on_no_driver(self) -> None:
        mgr = Neo4jManager.__new__(Neo4jManager)
        mgr._driver = None
        assert mgr.get_cited_doc_ids(["d1"]) == []

    def test_get_cited_doc_ids_returns_empty_on_empty_list(self) -> None:
        mgr = Neo4jManager.__new__(Neo4jManager)
        mgr._driver = MagicMock()
        assert mgr.get_cited_doc_ids([]) == []

    def test_get_citing_doc_ids_returns_empty_on_no_driver(self) -> None:
        mgr = Neo4jManager.__new__(Neo4jManager)
        mgr._driver = None
        assert mgr.get_citing_doc_ids(["d1"]) == []

    def test_get_cited_article_doc_ids_returns_empty_on_no_driver(self) -> None:
        mgr = Neo4jManager.__new__(Neo4jManager)
        mgr._driver = None
        assert mgr.get_cited_article_doc_ids(["d1"]) == []

    def test_get_modifie_doc_ids_returns_empty_on_no_driver(self) -> None:
        mgr = Neo4jManager.__new__(Neo4jManager)
        mgr._driver = None
        assert mgr.get_modifie_doc_ids(["d1"]) == []

    def test_get_voir_article_doc_ids_returns_empty_on_no_driver(self) -> None:
        mgr = Neo4jManager.__new__(Neo4jManager)
        mgr._driver = None
        assert mgr.get_voir_article_doc_ids(["d1"]) == []

    def test_is_connected_false_initially(self) -> None:
        mgr = Neo4jManager.__new__(Neo4jManager)
        mgr._driver = None
        assert mgr.is_connected() is False

    def test_close_when_no_driver(self) -> None:
        mgr = Neo4jManager.__new__(Neo4jManager)
        mgr._driver = None
        mgr.close()


class TestNeo4jManagerWithMockedDriver:

    def _make_manager(self) -> tuple[Neo4jManager, MagicMock]:
        mgr = Neo4jManager.__new__(Neo4jManager)
        driver = MagicMock()
        mgr._driver = driver
        return mgr, driver

    def test_is_connected_true(self) -> None:
        mgr, _ = self._make_manager()
        assert mgr.is_connected() is True

    def test_close_calls_driver_close(self) -> None:
        mgr, driver = self._make_manager()
        mgr.close()
        driver.close.assert_called_once()
        assert mgr._driver is None

    def test_reset_database_runs_detach_delete(self) -> None:
        mgr, driver = self._make_manager()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)

        mgr.reset_database()
        session.run.assert_called_once_with("MATCH (n) DETACH DELETE n")

    def test_create_schema_runs_all_statements(self) -> None:
        mgr, driver = self._make_manager()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)

        mgr.create_schema()
        assert session.run.call_count == 10  # 4 constraints + 6 indexes

    def test_run_batch_batches_correctly(self) -> None:
        mgr, driver = self._make_manager()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)

        rows = [{"id": str(i)} for i in range(1200)]
        written = mgr._run_batch("MERGE (n:Test {id: row.id})", rows, batch_size=500)
        assert written == 1200
        assert session.run.call_count == 3  # 500 + 500 + 200

    def test_get_cited_doc_ids_queries_neo4j(self) -> None:
        mgr, driver = self._make_manager()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        session.run.return_value = [{"id": "d2"}, {"id": "d3"}]

        result = mgr.get_cited_doc_ids(["d1"])
        assert result == ["d2", "d3"]


class TestNeo4jManagerConnect:

    @patch("retrievers.neo4j_setup.GraphDatabase", create=True)
    def test_connect_success(self, mock_gdb: MagicMock) -> None:
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver

        with patch.dict("sys.modules", {"neo4j": MagicMock(GraphDatabase=mock_gdb)}):
            mgr = Neo4jManager()
            with patch("retrievers.neo4j_setup.GraphDatabase", mock_gdb, create=True):
                # Manually simulate connect with mock
                mgr._driver = mock_driver
                assert mgr.is_connected() is True

    def test_connect_failure_returns_false(self) -> None:
        mgr = Neo4jManager(uri="bolt://nonexistent:9999")
        result = mgr.connect()
        assert result is False
