"""
LLM integration service for answer generation.

Uses Groq API for fast inference with Llama models.
"""

import logging
from typing import AsyncGenerator, Dict, List, Optional

from groq import AsyncGroq

from shared.models import SearchResult
from shared.utils import get_logger

logger = get_logger(__name__)


class LLMService:
    """
    LLM service for generating answers from retrieved context.

    Uses Groq API with Llama-3.3-70B for fast, high-quality responses.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        """
        Initialize the LLM service.

        Args:
            api_key: Groq API key
            model: Model name (llama-3.3-70b-versatile for speed)
            temperature: Sampling temperature (lower = more focused)
            max_tokens: Maximum response length
        """
        self.client = AsyncGroq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"Initialized LLM service with model '{model}'")

    async def generate_answer(
        self,
        query: str,
        context_chunks: List[SearchResult],
        stream: bool = False,
    ) -> str:
        """
        Generate an answer using retrieved context.

        Args:
            query: User's question
            context_chunks: Retrieved chunks for context
            stream: Whether to stream the response

        Returns:
            Generated answer text
        """
        # Build context from chunks
        context = self._build_context(context_chunks)

        # Build prompt
        prompt = self._build_prompt(query, context)

        logger.info(f"Generating answer for query: '{query}'")

        try:
            if stream:
                # Streaming not implemented in non-async context
                # Fall back to non-streaming
                pass

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content
            logger.info(f"Generated answer ({len(answer)} characters)")
            return answer

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    async def generate_answer_stream(
        self,
        query: str,
        context_chunks: List[SearchResult],
    ) -> AsyncGenerator[str, None]:
        """
        Generate an answer with streaming support.

        Args:
            query: User's question
            context_chunks: Retrieved chunks for context

        Yields:
            Answer text chunks as they are generated
        """
        context = self._build_context(context_chunks)
        prompt = self._build_prompt(query, context)

        logger.info(f"Generating streaming answer for query: '{query}'")

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"LLM streaming generation failed: {e}")
            raise

    def _build_context(self, chunks: List[SearchResult]) -> str:
        """
        Build context string from retrieved chunks.

        Args:
            chunks: Retrieved search results

        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context found."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            # Format: [Source 1] (Page 5) Content...
            source_info = f"[Source {i}]"
            if chunk.source_file:
                source_info += f" {chunk.source_file}"
            if chunk.page_numbers:
                pages = ", ".join(str(p) for p in chunk.page_numbers)
                source_info += f" (Page {pages})"
            if chunk.section_path:
                source_info += f" - {chunk.section_path}"

            context_parts.append(f"{source_info}\n{chunk.text}\n")

        return "\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build the user prompt with query and context.

        Args:
            query: User's question
            context: Formatted context from retrieval

        Returns:
            Complete prompt
        """
        return f"""Context from medical documents:

{context}

Question: {query}

Please provide a comprehensive, accurate answer based on the context above. If the context doesn't contain enough information to answer the question, clearly state that."""

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt defining assistant behavior.

        Returns:
            System prompt string
        """
        return """You are MedArchive AI, a clinical decision support assistant. Your role is to provide accurate, evidence-based information from medical guidelines and documentation.

Guidelines:
1. Answer based ONLY on the provided context - do not use external knowledge
2. If the context doesn't contain the answer, say so explicitly
3. Cite sources using [Source N] notation when referencing specific information
4. Use clear, professional medical language appropriate for healthcare providers
5. For dosing or treatment recommendations, include relevant warnings or contraindications mentioned in the context
6. If information is ambiguous or contradictory, acknowledge the contradiction
7. Structure longer answers with clear sections (e.g., Indication, Dosing, Monitoring)

Remember: Patient safety depends on accuracy. Never hallucinate or guess."""

    async def generate_standalone_question(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate a standalone question from a conversational query.

        Useful for handling follow-up questions that reference previous context.

        Args:
            query: Current user query
            conversation_history: Previous conversation turns

        Returns:
            Reformulated standalone question
        """
        if not conversation_history:
            return query

        # Build conversation context
        conv_text = "\n".join(
            f"{turn['role']}: {turn['content']}"
            for turn in conversation_history[-3:]  # Last 3 turns
        )

        prompt = f"""Given the conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that captures all necessary context.

Conversation history:
{conv_text}

Follow-up question: {query}

Standalone question:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You rephrase follow-up questions into standalone questions. Be concise and preserve all medical terminology.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.0,
                max_tokens=200,
            )

            standalone = response.choices[0].message.content.strip()
            logger.info(f"Reformulated query: '{query}' -> '{standalone}'")
            return standalone

        except Exception as e:
            logger.warning(f"Question reformulation failed, using original: {e}")
            return query
