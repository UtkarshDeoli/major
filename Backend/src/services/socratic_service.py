"""Socratic AI tutoring service.

Provides guided, step-by-step explanations without revealing the final answer
up front. Useful for "explain like a tutor" flows in chat and mock-test review.
"""
import json
from typing import Dict, Any, List, Optional
from fastapi import HTTPException

from src.services.gemini_service import gemini_service
from src.services.llm_service import _build_language_instruction


async def explain_socratically(
    question: str,
    concept: Optional[str] = None,
    doc_context: Optional[str] = None,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a Socratic breakdown of a question/concept.

    The response contains a series of hint/probe steps designed to guide the
    student toward the answer rather than giving it outright.
    """
    if not gemini_service:
        raise HTTPException(status_code=503, detail="Gemini service is not available")

    prompt_parts = [
        "You are a patient Socratic tutor. The student has asked a question. ",
        "DO NOT give the final answer in your first response. ",
        "Instead, guide the student through the reasoning with a short series of steps. ",
        "Each step should either: (1) ask a probing question, (2) give a small hint, ",
        "or (3) confirm partial understanding and nudge forward. ",
        "Respond ONLY with valid JSON in the exact format below.",
        "",
        f"Student question: {question}",
    ]
    if concept:
        prompt_parts.append(f"Specific concept to focus on: {concept}")
    if doc_context:
        prompt_parts.append(f"Relevant document context: {doc_context[:2000]}")

    language_instruction = _build_language_instruction(language)
    if language_instruction:
        prompt_parts.append(language_instruction)

    prompt_parts.extend([
        "",
        "Return JSON in this exact format:",
        json.dumps({
            "summary": "One-sentence framing of what we are exploring",
            "steps": [
                {
                    "type": "probe | hint | partial",
                    "content": "Step text: a question, hint, or partial explanation",
                    "expectation": "What the student should realize after this step"
                }
            ],
            "final_prompt": "A short closing question that checks if the student can now answer the original question"
        }, indent=2),
        "",
        "CRITICAL: Do not include the final answer in any step. Keep each step concise."
    ])

    prompt = "\n".join(prompt_parts)

    try:
        response = gemini_service.model.generate_content(prompt)
        if not response or not response.text:
            return _fallback_socratic(question)

        response_text = response.text.strip()
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        if start_idx == -1 or end_idx == 0:
            return _fallback_socratic(question)

        result = json.loads(response_text[start_idx:end_idx])
        return {
            "summary": result.get("summary", ""),
            "steps": result.get("steps", []),
            "final_prompt": result.get("final_prompt", "Can you now answer the original question?"),
        }
    except Exception as e:
        print(f"Socratic explanation error: {e}")
        return _fallback_socratic(question)


async def socratic_feedback_for_answer(
    question: str,
    user_answer: str,
    correct_answer: Optional[str] = None,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Give Socratic feedback on a specific student answer (right or wrong)."""
    if not gemini_service:
        raise HTTPException(status_code=503, detail="Gemini service is not available")

    prompt = f"""You are a Socratic tutor reviewing a student's answer.

Question: {question}
Student's answer: {user_answer}
{ f"Reference answer: {correct_answer}" if correct_answer else "" }

Respond ONLY with valid JSON in this exact format:
{{
    "is_on_track": true,
    "feedback_steps": [
        {{
            "type": "acknowledge | probe | hint | correction",
            "content": "Step text"
        }}
    ],
    "next_question": "A short follow-up question to deepen understanding"
}}

Guidelines:
- If the answer is correct, acknowledge briefly then ask a deeper follow-up.
- If the answer is wrong, identify the misconception gently and guide the student to discover the correction.
- Never be harsh.
- Keep each step concise.
"""
    language_instruction = _build_language_instruction(language)
    if language_instruction:
        prompt += f"\n\n{language_instruction}"

    try:
        response = gemini_service.model.generate_content(prompt)
        if not response or not response.text:
            return _fallback_feedback(user_answer)

        response_text = response.text.strip()
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        if start_idx == -1 or end_idx == 0:
            return _fallback_feedback(user_answer)

        result = json.loads(response_text[start_idx:end_idx])
        return {
            "is_on_track": result.get("is_on_track", True),
            "feedback_steps": result.get("feedback_steps", []),
            "next_question": result.get("next_question", ""),
        }
    except Exception as e:
        print(f"Socratic feedback error: {e}")
        return _fallback_feedback(user_answer)


def _fallback_socratic(question: str) -> Dict[str, Any]:
    return {
        "summary": f"Let's reason through this together: {question}",
        "steps": [
            {"type": "probe", "content": "What concepts do you think are most relevant here?", "expectation": "Identify key topics"},
            {"type": "hint", "content": "Try breaking the problem into smaller parts.", "expectation": "Outline a first step"},
            {"type": "partial", "content": "Consider the definitions involved and how they connect.", "expectation": "Relate definitions to the question"},
        ],
        "final_prompt": "What is your best attempt at the answer now?",
    }


def _fallback_feedback(user_answer: str) -> Dict[str, Any]:
    return {
        "is_on_track": bool(user_answer.strip()),
        "feedback_steps": [
            {"type": "acknowledge", "content": "Thanks for your answer."},
            {"type": "probe", "content": "What made you choose this reasoning?"},
        ],
        "next_question": "Can you explain your reasoning in another way?",
    }
