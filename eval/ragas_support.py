"""Helpers for optional Ragas metrics in the Veridicta evaluation pipeline."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Iterable

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision, Faithfulness

logger = logging.getLogger(__name__)

CEREBRAS_OPENAI_BASE_URL = "https://api.cerebras.ai/v1"
DEFAULT_RAGAS_BACKEND = "cerebras"
DEFAULT_RAGAS_LANGUAGE = "french"
DEFAULT_RAGAS_MODEL = "llama3.1-8b"
MAX_RAGAS_ANSWER_CHARS = 1_500
MAX_RAGAS_CHUNK_CHARS = 900
MAX_RAGAS_TOTAL_CONTEXT_CHARS = 3_600


class RagasConfigurationError(RuntimeError):
	"""Raised when the Ragas evaluator cannot be configured safely."""


@dataclass(frozen=True)
class RagasConfig:
	"""Runtime configuration for the optional Ragas judge."""

	backend: str = DEFAULT_RAGAS_BACKEND
	model: str = DEFAULT_RAGAS_MODEL
	language: str = DEFAULT_RAGAS_LANGUAGE


@dataclass(frozen=True)
class RagasScores:
	"""Normalized Ragas scores for a single evaluation sample."""

	faithfulness: float | None
	context_precision: float | None
	error: str | None = None


class RagasEvaluator:
	"""Thin wrapper around Ragas metric objects used by eval/evaluate.py."""

	def __init__(self, config: RagasConfig) -> None:
		self._config = config
		self._language = "english"
		self._llm = _build_ragas_llm(config)
		self._faithfulness = Faithfulness(llm=self._llm)
		self._context_precision = ContextPrecision(llm=self._llm)
		self._adapt_prompts_if_needed()

	@property
	def label(self) -> str:
		return f"{self._config.backend}/{self._config.model}"

	@property
	def language(self) -> str:
		return self._language

	def score(
		self,
		*,
		question: str,
		answer: str,
		reference_answer: str,
		retrieved_chunks: Iterable[dict],
	) -> RagasScores:
		sanitized_answer = _sanitize_answer(answer)
		contexts = _sanitize_contexts(retrieved_chunks)

		if not question.strip():
			return RagasScores(None, None, "missing_question")
		if not sanitized_answer:
			return RagasScores(None, None, "missing_answer")
		if not reference_answer.strip():
			return RagasScores(None, None, "missing_reference_answer")
		if not contexts:
			return RagasScores(None, None, "missing_retrieved_contexts")

		try:
			faithfulness_result = self._faithfulness.score(
				user_input=question,
				response=sanitized_answer,
				retrieved_contexts=contexts,
			)
			context_precision_result = self._context_precision.score(
				user_input=question,
				reference=reference_answer,
				retrieved_contexts=contexts,
			)
		except Exception as exc:  # pragma: no cover - network/runtime dependent
			logger.warning("Ragas scoring failed for question '%s': %s", question[:80], exc)
			return RagasScores(None, None, str(exc))

		return RagasScores(
			faithfulness=_normalise_metric_value(faithfulness_result.value),
			context_precision=_normalise_metric_value(context_precision_result.value),
		)

	def _adapt_prompts_if_needed(self) -> None:
		target_language = self._config.language.strip().lower()
		if not target_language or target_language == "english":
			return

		async def _adapt() -> None:
			self._faithfulness.statement_generator_prompt = (
				await self._faithfulness.statement_generator_prompt.adapt(
					target_language=target_language,
					llm=self._llm,
				)
			)
			self._faithfulness.nli_statement_prompt = (
				await self._faithfulness.nli_statement_prompt.adapt(
					target_language=target_language,
					llm=self._llm,
				)
			)
			self._context_precision.prompt = await self._context_precision.prompt.adapt(
				target_language=target_language,
				llm=self._llm,
			)

		logger.info("Adapting Ragas prompts to %s ...", target_language)
		try:
			asyncio.run(_adapt())
			self._language = target_language
		except Exception as exc:  # pragma: no cover - network/runtime dependent
			logger.warning(
				"Ragas prompt adaptation to %s failed, continuing with English prompts: %s",
				target_language,
				exc,
			)


def _build_ragas_llm(config: RagasConfig):
	backend = config.backend.strip().lower()
	if backend != DEFAULT_RAGAS_BACKEND:
		raise RagasConfigurationError(
			"Only the Cerebras judge is currently supported for Ragas. "
			"Use --ragas-backend cerebras."
		)

	api_key = os.getenv("CEREBRAS_API_KEY")
	if not api_key:
		raise RagasConfigurationError(
			"CEREBRAS_API_KEY not set. Ragas needs a judge model configured in the environment."
		)

	client = AsyncOpenAI(api_key=api_key, base_url=CEREBRAS_OPENAI_BASE_URL)
	return llm_factory(config.model, provider="openai", client=client)


def _normalise_metric_value(value: float | None) -> float | None:
	if value is None:
		return None
	numeric_value = float(value)
	if math.isnan(numeric_value):
		return None
	return round(numeric_value, 4)


def _sanitize_answer(answer: str) -> str:
	answer_without_citations = re.sub(r"\[Source\s+\d+\]", "", answer)
	return _collapse_whitespace(answer_without_citations)[:MAX_RAGAS_ANSWER_CHARS].strip()


def _sanitize_contexts(retrieved_chunks: Iterable[dict]) -> list[str]:
	sanitized_contexts: list[str] = []
	total_chars = 0

	for chunk in retrieved_chunks:
		text = _collapse_whitespace(chunk.get("text", ""))
		if not text:
			continue

		remaining_chars = MAX_RAGAS_TOTAL_CONTEXT_CHARS - total_chars
		if remaining_chars <= 0:
			break

		trimmed_text = text[: min(MAX_RAGAS_CHUNK_CHARS, remaining_chars)].strip()
		if not trimmed_text:
			continue

		sanitized_contexts.append(trimmed_text)
		total_chars += len(trimmed_text)

	return sanitized_contexts


def _collapse_whitespace(text: str) -> str:
	return re.sub(r"\s+", " ", text).strip()
