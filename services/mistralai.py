import os
import json
from mistralai import Mistral
from config.logger import logger
from transformers import pipeline


# ========== Existing Grammar Functions ==========

def grammar_corrector_t5():
    """
    Uses T5 model for grammar correction - runs locally, no API key needed
    """

    corrector = pipeline(
        "text2text-generation",
        model="vennify/t5-base-grammar-correction"
    )
    return corrector


def correct_text(corrector, text):
    """Correct grammar in the given text"""
    result = corrector(text, max_length=512, num_beams=4)
    return result[0]['generated_text']


def correct_grammar_service(prompt: str):
    corrector = grammar_corrector_t5()
    corrected_text = correct_text(corrector, prompt)
    return corrected_text


# ========== Existing Mistral Service ==========

def get_mistralai_service(prompt: str, model: str = "mistral-small-latest") -> dict:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        logger.error("MISTRAL_API_KEY is not set in environment variables.")
        raise EnvironmentError("Missing MISTRAL_API_KEY environment variable")

    client = Mistral(api_key=api_key)

    chat_response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    return chat_response.choices[0].message.content


# ========== NEW SERVICE: Query + Retrieval Context ==========
def get_mistral_with_context_service(query: str, context_list: list, model: str = "mistral-large-latest") -> str:
    """
    Pass a user query and a list of dicts (retrieved data) to Mistral for contextual answers.
    """
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        logger.error("MISTRAL_API_KEY is not set in environment variables.")
        raise EnvironmentError("Missing MISTRAL_API_KEY environment variable")

    client = Mistral(api_key=api_key)

    # Format list of dicts into readable bullet points for context
    formatted_context = ""
    for i, item in enumerate(context_list, 1):
        formatted_context += f"\nContext {i}:\n"
        for key, value in item.items():
            formatted_context += f"- {key}: {value}\n"

    prompt = f"""
You are an intelligent assistant. 
Use the provided contexts below to answer the user's query accurately.
And provide personilized response based on the me data.
{formatted_context}

User Query:
{query}

If the answer is not found in the provided contexts, respond with:
"I don't have enough information in the provided context."
And dont send reponse in rich test like ** and all
"""

    chat_response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = chat_response.choices[0].message.content.strip()
    return response_text
