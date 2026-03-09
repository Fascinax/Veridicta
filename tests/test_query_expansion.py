from eval.evaluate import _expand_query_legal_fr


def test_expand_query_legal_fr_adds_expected_terms() -> None:
    question = "Quelles sont les indemnités de licenciement en cas de faute grave ?"

    expanded = _expand_query_legal_fr(question)

    assert "congediement" in expanded
    assert "preavis" in expanded
    assert "cause valable" in expanded


def test_expand_query_legal_fr_keeps_unmatched_question_unchanged() -> None:
    question = "Quel est le code postal de Monaco ?"

    expanded = _expand_query_legal_fr(question)

    assert expanded == question
