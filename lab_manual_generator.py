"""
lab_manual_generator.py — Step-by-step lab manual generation engine.

Uses IBM Granite via RAG to produce complete, educator-ready lab manuals
including objectives, theory, procedures, materials lists, and safety notes.
"""

from __future__ import annotations

import logging
from typing import Any

from config import Config
from curriculum_retriever import get_course_context, search_curriculum
from rag_pipeline import rag_generate

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# System instruction for manual generation
# ─────────────────────────────────────────────────────────────────────────────

_MANUAL_SYSTEM_INSTRUCTION = """You are an expert academic lab manual author for STEM education.
You create comprehensive, safe, and pedagogically sound laboratory manuals.
Always include: clear learning objectives, background theory, required materials with quantities,
step-by-step numbered procedure, safety precautions, data recording tables, and discussion questions.
Base your content strictly on the retrieved curriculum context provided.
Format the output clearly with section headers using markdown (##, ###).
Use precise scientific language appropriate for undergraduate students."""


# ─────────────────────────────────────────────────────────────────────────────
# Section-level prompt templates
# ─────────────────────────────────────────────────────────────────────────────

def _build_manual_prompt(
    topic: str,
    course_code: str,
    equipment_available: list[str],
    duration_hours: float,
    student_level: str,
) -> str:
    equip_str = ", ".join(equipment_available) if equipment_available else "standard laboratory equipment"
    return (
        f"Generate a complete laboratory manual for the following experiment:\n"
        f"- Course Code: {course_code}\n"
        f"- Experiment Topic: {topic}\n"
        f"- Available Equipment: {equip_str}\n"
        f"- Lab Duration: {duration_hours} hour(s)\n"
        f"- Student Level: {student_level}\n\n"
        f"The manual must include ALL of the following sections:\n"
        f"1. Title and Course Information\n"
        f"2. Learning Objectives (3-5 measurable objectives)\n"
        f"3. Background Theory (200-300 words)\n"
        f"4. Materials and Equipment (itemised list with quantities)\n"
        f"5. Safety Precautions (minimum 5 specific precautions)\n"
        f"6. Pre-Lab Questions (3 questions)\n"
        f"7. Step-by-Step Procedure (numbered steps, minimum 10 steps)\n"
        f"8. Data Recording Tables (formatted tables for measurements)\n"
        f"9. Sample Calculations\n"
        f"10. Discussion Questions (5 questions)\n"
        f"11. References and Further Reading\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_lab_manual(
    topic: str,
    course_code: str = "GENERAL",
    equipment_available: list[str] | None = None,
    duration_hours: float = 3.0,
    student_level: str = "undergraduate",
) -> dict[str, Any]:
    """
    Generate a complete lab manual for *topic* in *course_code*.

    Parameters
    ----------
    topic : str
        The experiment topic (e.g. "Acid-Base Titration").
    course_code : str
        Course identifier (e.g. "CHM101").
    equipment_available : list[str] or None
        Equipment present in the lab; None means use curriculum defaults.
    duration_hours : float
        Estimated lab session duration in hours.
    student_level : str
        Educational level (e.g. "undergraduate", "high school").

    Returns
    -------
    dict with keys: ``manual_text``, ``course_code``, ``topic``,
    ``retrieved_chunks``, ``word_count``.
    """
    equip = equipment_available or []
    course_context = get_course_context(course_code)
    query = _build_manual_prompt(topic, course_code, equip, duration_hours, student_level)

    logger.info("Generating lab manual: course=%s, topic=%s", course_code, topic)

    result = rag_generate(
        query=query,
        system_instruction=_MANUAL_SYSTEM_INSTRUCTION,
        extra_context=course_context,
        max_new_tokens=1500,
    )

    manual_text = result["answer"]
    word_count = len(manual_text.split())

    return {
        "manual_text": manual_text,
        "course_code": course_code,
        "topic": topic,
        "duration_hours": duration_hours,
        "student_level": student_level,
        "retrieved_chunks": result["retrieved_chunks"],
        "word_count": word_count,
    }


def generate_safety_instructions(
    topic: str,
    course_code: str = "GENERAL",
    chemicals: list[str] | None = None,
    equipment: list[str] | None = None,
) -> dict[str, Any]:
    """
    Generate targeted safety instructions for a specific experiment.
    """
    chem_str = ", ".join(chemicals) if chemicals else "standard reagents"
    equip_str = ", ".join(equipment) if equipment else "standard equipment"
    query = (
        f"Generate detailed safety instructions for a {topic} experiment in {course_code}.\n"
        f"Chemicals/Reagents: {chem_str}\n"
        f"Equipment: {equip_str}\n"
        f"Include: PPE requirements, chemical hazards, first aid procedures, "
        f"waste disposal, emergency procedures, and a pre-lab safety checklist."
    )

    system = (
        "You are a certified laboratory safety officer. "
        "Generate clear, specific, and comprehensive safety instructions for STEM laboratories. "
        "Prioritise student safety. Use bullet points and bold critical warnings."
    )

    result = rag_generate(
        query=query,
        system_instruction=system,
        extra_context=get_course_context(course_code),
    )

    return {
        "safety_text": result["answer"],
        "course_code": course_code,
        "topic": topic,
        "retrieved_chunks": result["retrieved_chunks"],
    }
