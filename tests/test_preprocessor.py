"""Testes unitários para o módulo src/preprocessor.py."""

import pytest
from src.preprocessor import (
    TextPreprocessor,
    CONFIG_NO_STOP_NO_STEM,
    CONFIG_WITH_STOP_NO_STEM,
    CONFIG_NO_STOP_WITH_STEM,
    CONFIG_WITH_STOP_WITH_STEM,
    preprocess_corpus,
)


SAMPLE_TEXT = "The experimental investigations of the aerodynamics were conducted in 1958."


def test_tokenization_and_lowercasing():
    prep = TextPreprocessor(CONFIG_NO_STOP_NO_STEM)
    tokens = prep.process(SAMPLE_TEXT)
    assert "the" in tokens
    assert "investigations" in tokens
    assert "1958" in tokens
    assert "." not in tokens
    assert len(tokens) == 10


def test_stopwords_removal():
    prep_no_stop = TextPreprocessor(CONFIG_NO_STOP_NO_STEM)
    prep_with_stop = TextPreprocessor(CONFIG_WITH_STOP_NO_STEM)

    tokens_raw = prep_no_stop.process(SAMPLE_TEXT)
    tokens_filtered = prep_with_stop.process(SAMPLE_TEXT)

    for sw in ["the", "of", "were", "in"]:
        assert sw in tokens_raw
        assert sw not in tokens_filtered

    assert "experimental" in tokens_filtered
    assert "investigations" in tokens_filtered
    assert "aerodynamics" in tokens_filtered


def test_stemming():
    prep_no_stem = TextPreprocessor(CONFIG_NO_STOP_NO_STEM)
    prep_with_stem = TextPreprocessor(CONFIG_NO_STOP_WITH_STEM)

    tokens_no_stem = prep_no_stem.process(SAMPLE_TEXT)
    tokens_stemmed = prep_with_stem.process(SAMPLE_TEXT)

    assert "investigations" in tokens_no_stem
    assert "investig" in tokens_stemmed
    assert "investigations" not in tokens_stemmed
    assert "conduct" in tokens_stemmed


def test_combined_stopwords_and_stemming():
    prep_full = TextPreprocessor(CONFIG_WITH_STOP_WITH_STEM)
    tokens = prep_full.process(SAMPLE_TEXT)

    assert "the" not in tokens
    assert "of" not in tokens
    assert "were" not in tokens

    assert "investig" in tokens
    assert "conduct" in tokens
    assert "aerodynam" in tokens


def test_empty_and_special_cases():
    prep = TextPreprocessor(CONFIG_WITH_STOP_WITH_STEM)
    assert prep.process("") == []
    assert prep.process("   \n\t  ") == []
    assert prep.process("!!! ??? ...") == []


def test_preprocess_corpus_helper():
    corpus = {
        "d1": "Boundary layer transition.",
        "d2": "Heat transfer in supersonic flow.",
    }
    result = preprocess_corpus(corpus, CONFIG_WITH_STOP_WITH_STEM)
    assert len(result) == 2
    assert "boundari" in result["d1"]
    assert "heat" in result["d2"]
