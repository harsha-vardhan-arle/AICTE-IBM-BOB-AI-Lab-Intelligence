"""
rubric_generator.py — Evaluation rubric generation engine.

Generates detailed grading rubrics aligned with educational standards,
learning objectives, and institutional assessment guidelines.
Supports multiple rubric types: holistic, analytic, single-point.
"""

from __future__ import annotations

import logging
from typing import Any

from config import Config
from curriculum_retriever import get_course_context
from rag_pipeline import rag_generate

logger = logging.getLogger(__name__)


_RUBRIC_SYSTEM = """You are an expert in educational assessment and rubric design for STEM lab courses.
Create detailed, fair, and measurable evaluation rubrics aligned with Bloom's Taxonomy and learning outcomes.
Use clear performance descriptors for each criterion and score band.
Always include: criterion name, weight/points, and descriptions for Excellent/Good/Satisfactory/Needs Improvement.
Format as a structured markdown table for each rubric criterion."""


RUBRIC_TEMPLATES = {
    "lab_report": {
        "total_points": 100,
        "criteria": [
            {"name": "Pre-Lab Preparation", "points": 15, "weight_pct": 15},
            {"name": "Safety Compliance", "points": 10, "weight_pct": 10},
            {"name": "Experimental Procedure", "points": 25, "weight_pct": 25},
            {"name": "Data Collection & Recording", "points": 20, "weight_pct": 20},
            {"name": "Analysis & Calculations", "points": 20, "weight_pct": 20},
            {"name": "Conclusion & Reflection", "points": 10, "weight_pct": 10},
        ],
    },
    "virtual_lab": {
        "total_points": 100,
        "criteria": [
            {"name": "Simulation Setup", "points": 10, "weight_pct": 10},
            {"name": "Procedure Execution", "points": 25, "weight_pct": 25},
            {"name": "Data Collection & Screenshot Evidence", "points": 20, "weight_pct": 20},
            {"name": "Analysis & Interpretation", "points": 25, "weight_pct": 25},
            {"name": "Report Quality & Clarity", "points": 20, "weight_pct": 20},
        ],
    },
    "practical_exam": {
        "total_points": 100,
        "criteria": [
            {"name": "Technique & Skill", "points": 30, "weight_pct": 30},
            {"name": "Safety Practices", "points": 15, "weight_pct": 15},
            {"name": "Accuracy of Results", "points": 30, "weight_pct": 30},
            {"name": "Time Management", "points": 10, "weight_pct": 10},
            {"name": "Lab Notebook / Documentation", "points": 15, "weight_pct": 15},
        ],
    },
    "group_project": {
        "total_points": 100,
        "criteria": [
            {"name": "Research & Background", "points": 20, "weight_pct": 20},
            {"name": "Experimental Design", "points": 20, "weight_pct": 20},
            {"name": "Data Analysis & Results", "points": 25, "weight_pct": 25},
            {"name": "Presentation & Communication", "points": 20, "weight_pct": 20},
            {"name": "Team Collaboration", "points": 15, "weight_pct": 15},
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_evaluation_rubric(
    topic: str,
    course_code: str = "GENERAL",
    rubric_type: str = "lab_report",
    learning_objectives: list[str] | None = None,
    total_points: int = 100,
    custom_criteria: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Generate a complete evaluation rubric for a lab experiment or assignment.

    Parameters
    ----------
    topic : str
        The experiment/assignment being assessed.
    course_code : str
        Course identifier.
    rubric_type : str
        One of: ``lab_report``, ``virtual_lab``, ``practical_exam``, ``group_project``.
    learning_objectives : list[str] or None
        Specific learning objectives the rubric should assess.
    total_points : int
        Total point value for the rubric.
    custom_criteria : list[dict] or None
        Override default criteria with custom ones.

    Returns
    -------
    dict with keys: ``rubric_text``, ``rubric_structure``, ``criteria``, ``total_points``.
    """
    template = RUBRIC_TEMPLATES.get(rubric_type, RUBRIC_TEMPLATES["lab_report"])
    criteria = custom_criteria or template["criteria"]

    objectives_str = (
        "\n".join(f"  - {obj}" for obj in learning_objectives)
        if learning_objectives
        else "  - As defined in the course curriculum"
    )
    criteria_str = "\n".join(
        f"  - {c['name']} ({c['points']} pts)" for c in criteria
    )

    query = (
        f"Generate a detailed evaluation rubric for: {topic}\n"
        f"Course: {course_code}\n"
        f"Rubric Type: {rubric_type.replace('_', ' ').title()}\n"
        f"Total Points: {total_points}\n"
        f"Learning Objectives to Assess:\n{objectives_str}\n"
        f"Rubric Criteria:\n{criteria_str}\n\n"
        f"For each criterion, provide a markdown table with columns: "
        f"Criterion | Points | Excellent (90-100%) | Good (75-89%) | Satisfactory (60-74%) | Needs Improvement (<60%)\n"
        f"After all criteria tables, include: "
        f"1. Total points summary, "
        f"2. Grade conversion scale, "
        f"3. Instructor notes for consistent grading."
    )

    result = rag_generate(
        query=query,
        system_instruction=_RUBRIC_SYSTEM,
        extra_context=get_course_context(course_code),
        max_new_tokens=1400,
    )

    return {
        "rubric_text": result["answer"],
        "course_code": course_code,
        "topic": topic,
        "rubric_type": rubric_type,
        "criteria": criteria,
        "total_points": total_points,
        "learning_objectives": learning_objectives or [],
        "retrieved_chunks": result["retrieved_chunks"],
    }


def generate_grade_conversion_scale(
    total_points: int = 100,
    grading_system: str = "letter",
) -> dict[str, Any]:
    """
    Generate a grade conversion scale for a given point total.

    grading_system: 'letter' (A/B/C/D/F), 'percentage', or 'gpa'
    """
    if grading_system == "letter":
        bands = [
            {"grade": "A", "min_pct": 90, "min_pts": int(total_points * 0.90), "description": "Excellent"},
            {"grade": "B", "min_pct": 80, "min_pts": int(total_points * 0.80), "description": "Good"},
            {"grade": "C", "min_pct": 70, "min_pts": int(total_points * 0.70), "description": "Satisfactory"},
            {"grade": "D", "min_pct": 60, "min_pts": int(total_points * 0.60), "description": "Passing (Minimum)"},
            {"grade": "F", "min_pct": 0, "min_pts": 0, "description": "Failing"},
        ]
    elif grading_system == "gpa":
        bands = [
            {"grade": "4.0 (A)", "min_pct": 93, "min_pts": int(total_points * 0.93), "description": "Outstanding"},
            {"grade": "3.7 (A-)", "min_pct": 90, "min_pts": int(total_points * 0.90), "description": "Excellent"},
            {"grade": "3.3 (B+)", "min_pct": 87, "min_pts": int(total_points * 0.87), "description": "Very Good"},
            {"grade": "3.0 (B)", "min_pct": 83, "min_pts": int(total_points * 0.83), "description": "Good"},
            {"grade": "2.7 (B-)", "min_pct": 80, "min_pts": int(total_points * 0.80), "description": "Above Average"},
            {"grade": "2.3 (C+)", "min_pct": 77, "min_pts": int(total_points * 0.77), "description": "Average"},
            {"grade": "2.0 (C)", "min_pct": 73, "min_pts": int(total_points * 0.73), "description": "Satisfactory"},
            {"grade": "0.0 (F)", "min_pct": 0, "min_pts": 0, "description": "Failing"},
        ]
    else:
        bands = [
            {"grade": f"{p}%", "min_pct": p, "min_pts": int(total_points * p / 100), "description": ""}
            for p in [90, 80, 70, 60, 50, 0]
        ]

    return {
        "grading_system": grading_system,
        "total_points": total_points,
        "bands": bands,
    }


def get_available_rubric_types() -> list[str]:
    """Return the names of all available rubric templates."""
    return list(RUBRIC_TEMPLATES.keys())
