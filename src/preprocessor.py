"""Módulo de pré-processamento textual (tokenização, stopwords e stemming)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import PorterStemmer


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuração de pré-processamento."""
    remove_stopwords: bool
    apply_stemming: bool
    name: str

    @property
    def label(self) -> str:
        stop_str = "Com Stopwords" if self.remove_stopwords else "Sem Remoção Stopwords"
        stem_str = "Com Stemming" if self.apply_stemming else "Sem Stemming"
        return f"{stop_str} | {stem_str}"


# Configurações de pré-processamento
CONFIG_NO_STOP_NO_STEM = PreprocessingConfig(
    remove_stopwords=False, apply_stemming=False, name="no_stop_no_stem"
)
CONFIG_WITH_STOP_NO_STEM = PreprocessingConfig(
    remove_stopwords=True, apply_stemming=False, name="with_stop_no_stem"
)
CONFIG_NO_STOP_WITH_STEM = PreprocessingConfig(
    remove_stopwords=False, apply_stemming=True, name="no_stop_with_stem"
)
CONFIG_WITH_STOP_WITH_STEM = PreprocessingConfig(
    remove_stopwords=True, apply_stemming=True, name="with_stop_with_stem"
)

PREPROCESSING_CONFIGS: List[PreprocessingConfig] = [
    CONFIG_NO_STOP_NO_STEM,
    CONFIG_WITH_STOP_NO_STEM,
    CONFIG_NO_STOP_WITH_STEM,
    CONFIG_WITH_STOP_WITH_STEM,
]


class TextPreprocessor:
    """
    Classe responsável pelo pipeline de pré-processamento de texto.
    """

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self._stemmer = PorterStemmer() if config.apply_stemming else None
        
        if config.remove_stopwords:
            try:
                self._stopwords: Set[str] = set(nltk_stopwords.words("english"))
            except LookupError:
                nltk.download("stopwords", quiet=True)
                self._stopwords = set(nltk_stopwords.words("english"))
        else:
            self._stopwords = set()

        self._token_pattern = re.compile(r"\b[a-zA-Z0-9]+(?:'[a-zA-Z]+)?\b")

    def tokenize(self, text: str) -> List[str]:
        """
        Converte o texto para minúsculas e extrai tokens alfanuméricos.
        """
        if not text:
            return []
        lowered = text.lower()
        return self._token_pattern.findall(lowered)

    def filter_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords da lista de tokens se configurado.
        """
        if not self.config.remove_stopwords:
            return tokens
        return [t for t in tokens if t not in self._stopwords]

    def stem(self, tokens: List[str]) -> List[str]:
        """
        Aplica o algoritmo PorterStemmer aos tokens se configurado.
        """
        if not self.config.apply_stemming or self._stemmer is None:
            return tokens
        return [self._stemmer.stem(t) for t in tokens]

    def process(self, text: str) -> List[str]:
        """Executa tokenização, remoção de stopwords e stemming conforme a configuração."""
        tokens = self.tokenize(text)
        if self.config.remove_stopwords:
            tokens = self.filter_stopwords(tokens)
        if self.config.apply_stemming:
            tokens = self.stem(tokens)
        return tokens

    def process_to_string(self, text: str) -> str:
        """Retorna os tokens processados unidos por espaço."""
        return " ".join(self.process(text))


def preprocess_corpus(
    texts: Dict[str, str], config: PreprocessingConfig
) -> Dict[str, List[str]]:
    """Aplica o pré-processamento a um dicionário de textos (id -> tokens)."""
    preprocessor = TextPreprocessor(config)
    return {doc_id: preprocessor.process(text) for doc_id, text in texts.items()}


def preprocess_corpus_to_strings(
    texts: Dict[str, str], config: PreprocessingConfig
) -> Dict[str, str]:
    """
    Aplica o pré-processamento a um dicionário de textos, retornando strings limpas.
    """
    preprocessor = TextPreprocessor(config)
    return {doc_id: preprocessor.process_to_string(text) for doc_id, text in texts.items()}
