import os
from groq import Groq
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


# This helper function is now more of a general safeguard,
# but the core logic is handled inside generate_answer.
def limit_model_input(input_str: str, max_len: int = 44000):
    """Truncates the input string to a maximum length."""
    if len(input_str) > max_len:
        return input_str[:max_len]
    return input_str


class GroqLLMClient:
    def __init__(
        self,
        api_key: str,
        model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    ):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def generate_multi_query(self, original_query: str) -> str:
        """Generate an alternative query for multi-query retrieval."""
        prompt = f"""Given the following user question, generate ONE alternative question that asks for the same information but uses different words and phrasing. The alternative question should help retrieve relevant documents that might not match the original query exactly.

Original question: {original_query}

Alternative question:"""

        try:
            # Using limit_model_input here is fine as the prompt is small.
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates alternative search queries.",
                    },
                    # A smaller limit is more than enough for this task.
                    {
                        "role": "user",
                        "content": limit_model_input(prompt, max_len=2000),
                    },
                ],
                model=self.model_name,
                max_tokens=100,
                temperature=0.7,
            )

            alternative_query = response.choices[0].message.content.strip()
            logger.info(f"Generated alternative query: {alternative_query}")
            return alternative_query

        except Exception as e:
            logger.error(f"Failed to generate alternative query: {e}")
            return original_query  # Fallback to original query

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer based on query and retrieved context chunks."""

        # ---- START: Revised Logic ----

        # Define a maximum character limit for the entire prompt payload.
        # This should be less than the model's absolute max to leave room for variability.
        MAX_PROMPT_LEN = 44000

        # 1. Define the prompt structure with placeholders for context and query.
        prompt_template = """Based on the provided context, answer the following question. If the context doesn't contain enough information to answer the question completely, say so and provide what information you can find.

Context:
{context_text}

Question: {query}

Answer:"""

        # 2. Calculate the length of the prompt's boilerplate (everything EXCEPT the context).
        boilerplate_prompt = prompt_template.format(context_text="", query=query)
        boilerplate_len = len(boilerplate_prompt)

        # 3. Calculate the maximum character length available for the context.
        max_len_for_context = MAX_PROMPT_LEN - boilerplate_len

        # 4. Iteratively build the context, stopping when it's full.
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            header = chunk.get("contextual_header", "")
            text = chunk.get("text", "")
            chunk_str = f"[Source {i}] {header}\n{text}\n\n"

            # If adding the next chunk exceeds the available space, stop.
            if len(context_text) + len(chunk_str) > max_len_for_context:
                logger.warning(
                    f"Truncating context after {i-1} chunks to fit model input limits."
                )
                break

            context_text += chunk_str

        # 5. Assemble the final prompt with the (potentially truncated) context.
        final_prompt = prompt_template.format(context_text=context_text, query=query)

        # ---- END: Revised Logic ----

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided context. Always cite which sources you used in your answer using [Source X] format.",
                    },
                    # The final_prompt is now guaranteed to be within the size limit.
                    {"role": "user", "content": final_prompt},
                ],
                model=self.model_name,
                max_tokens=2000,
                temperature=0.3,
            )

            answer = response.choices[0].message.content.strip()
            logger.info("Generated answer successfully")
            return answer

        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return "I apologize, but I encountered an error while generating the answer. Please try again."
