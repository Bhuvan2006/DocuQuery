SYSTEM_PROMPT = """
You are a clinical decision support assistant.

You must explain a diabetes risk assessment using ONLY the structured data provided.
You are NOT allowed to:
- Add new risk factors
- Add medical advice
- Diagnose disease
- Infer causes not explicitly mentioned
- Mention model internals or algorithms

You MUST:
- Use cautious, non-diagnostic language
- Explain why the risk level was assigned
- Reference only the provided factors and patterns
- Write for clinicians (clear, concise, neutral tone)

If information is missing, do NOT speculate.
"""


def build_user_prompt(reasoning_json: dict) -> str:
    return f"""
Using the following structured risk assessment data, explain the result
in clear, clinician-friendly language.

Risk Assessment Data (GROUND TRUTH):
{reasoning_json}

Do not introduce new information.
"""

from openai import OpenAI
import json

client = OpenAI(api_key="YOUR_API_KEY")

def generate_explanation(reasoning_json: dict) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or gpt-4.1, or local model
        temperature=0.2,      # LOW temperature = less hallucination
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(reasoning_json)}
        ]
    )

    return response.choices[0].message.content
