"""
experiment_simulator.py — Virtual experiment simulation generator.

Generates step-by-step virtual/simulation experiment descriptions including
expected observations, interactive decision points, and virtual data sets.
Supports remote lab scenarios using approved simulation platforms.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from config import Config
from curriculum_retriever import get_course_context
from rag_pipeline import rag_generate

logger = logging.getLogger(__name__)


_SIMULATION_SYSTEM = """You are an expert in virtual laboratory education and simulation design.
Generate detailed, realistic virtual experiment simulations that mirror physical lab experiences.
Include: simulation setup instructions, expected observations at each step, what-if scenarios,
common mistakes students make and how the simulation shows their consequences,
and links to recommended virtual simulation tools.
Format with clear sections and numbered steps. Be specific about expected data values and ranges."""

_VIRTUAL_PLATFORMS = {
    "physics": ["PhET Interactive Simulations (phet.colorado.edu)", "Algodoo", "Tracker Video Analysis"],
    "chemistry": ["PhET Interactive Simulations", "Labster", "ChemCollective Virtual Lab"],
    "biology": ["Labster", "BioDigital Human", "HHMI BioInteractive"],
    "computer_science": ["Replit", "Jupyter Notebook", "CodePen"],
    "electrical_engineering": ["CircuitLab", "Tinkercad Circuits", "Falstad Circuit Simulator"],
    "environmental_science": ["GLOBE Observer App", "ArcGIS Online", "EPA BASINS"],
    "mathematics": ["GeoGebra", "Wolfram Alpha", "Desmos"],
}


def _get_platform_suggestions(subject: str) -> list[str]:
    """Return recommended virtual platforms for a given subject."""
    key = subject.lower().replace(" ", "_")
    for k, platforms in _VIRTUAL_PLATFORMS.items():
        if k in key or key in k:
            return platforms
    return ["PhET Interactive Simulations", "Labster", "Wolfram Alpha"]


def _generate_virtual_data(experiment_type: str, num_trials: int = 5) -> list[dict[str, Any]]:
    """
    Generate realistic simulated data sets for common experiment types.
    Returns structured trial data with slight random variation.
    """
    data_templates: dict[str, Any] = {
        "titration": {
            "columns": ["Trial", "Initial_Burette_Reading_mL", "Final_Burette_Reading_mL", "Volume_Used_mL", "Color_Change"],
            "generator": lambda t: {
                "Trial": t,
                "Initial_Burette_Reading_mL": round(random.uniform(0.0, 1.0), 2),
                "Final_Burette_Reading_mL": round(random.uniform(24.5, 25.5), 2),
                "Volume_Used_mL": round(random.uniform(24.0, 25.5), 2),
                "Color_Change": "Pale pink (endpoint)",
            },
        },
        "ohms_law": {
            "columns": ["Trial", "Voltage_V", "Current_mA", "Resistance_Ohm"],
            "generator": lambda t: {
                "Trial": t,
                "Voltage_V": round(t * 1.5, 1),
                "Current_mA": round((t * 1.5 / 100) * 1000 + random.uniform(-1, 1), 1),
                "Resistance_Ohm": 100,
            },
        },
        "pendulum": {
            "columns": ["Trial", "Length_cm", "Period_s", "g_measured_ms2"],
            "generator": lambda t: {
                "Trial": t,
                "Length_cm": t * 10,
                "Period_s": round(2 * 3.14159 * ((t * 0.1) / 9.81) ** 0.5 + random.uniform(-0.02, 0.02), 3),
                "g_measured_ms2": round(9.81 + random.uniform(-0.1, 0.1), 2),
            },
        },
        "default": {
            "columns": ["Trial", "Independent_Variable", "Dependent_Variable", "Notes"],
            "generator": lambda t: {
                "Trial": t,
                "Independent_Variable": round(t * 2.5, 2),
                "Dependent_Variable": round(t * 2.5 * 1.8 + random.uniform(-0.5, 0.5), 3),
                "Notes": "Within expected range",
            },
        },
    }

    key = next(
        (k for k in data_templates if k in experiment_type.lower().replace(" ", "_")),
        "default",
    )
    template = data_templates[key]
    return {
        "columns": template["columns"],
        "rows": [template["generator"](t + 1) for t in range(num_trials)],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_virtual_experiment(
    topic: str,
    course_code: str = "GENERAL",
    subject: str = "General",
    num_trials: int = 5,
    include_virtual_data: bool = True,
) -> dict[str, Any]:
    """
    Generate a complete virtual experiment simulation guide.

    Parameters
    ----------
    topic : str
        Experiment topic (e.g. "Simple Pendulum").
    course_code : str
        Course identifier.
    subject : str
        Subject area for platform recommendations.
    num_trials : int
        Number of simulated data trials to generate.
    include_virtual_data : bool
        Whether to include a sample simulated dataset.

    Returns
    -------
    dict with keys: ``simulation_guide``, ``virtual_platforms``,
    ``sample_data``, ``retrieved_chunks``.
    """
    platforms = _get_platform_suggestions(subject)
    platforms_str = "; ".join(platforms)
    course_ctx = get_course_context(course_code)

    query = (
        f"Create a complete virtual experiment simulation guide for: {topic}\n"
        f"Course: {course_code} ({subject})\n"
        f"Recommended Simulation Platforms: {platforms_str}\n\n"
        f"Include:\n"
        f"1. Simulation Setup Instructions (step-by-step how to open and configure the simulation)\n"
        f"2. Virtual Equipment Setup (what virtual instruments to place and configure)\n"
        f"3. Step-by-Step Virtual Procedure ({num_trials} trial runs)\n"
        f"4. Expected Observations (what students should see at each stage)\n"
        f"5. What-If Scenarios (3 scenarios exploring how changes affect results)\n"
        f"6. Common Virtual Lab Mistakes and How to Recognise Them\n"
        f"7. Data Export and Analysis Instructions\n"
        f"8. Comparison: Virtual vs Physical Lab (key differences and limitations)\n"
    )

    result = rag_generate(
        query=query,
        system_instruction=_SIMULATION_SYSTEM,
        extra_context=course_ctx,
        max_new_tokens=1400,
    )

    sample_data = _generate_virtual_data(topic, num_trials) if include_virtual_data else None

    return {
        "simulation_guide": result["answer"],
        "course_code": course_code,
        "topic": topic,
        "subject": subject,
        "virtual_platforms": platforms,
        "sample_data": sample_data,
        "retrieved_chunks": result["retrieved_chunks"],
    }


def generate_what_if_scenarios(
    topic: str,
    course_code: str = "GENERAL",
    variable: str = "temperature",
    num_scenarios: int = 3,
) -> dict[str, Any]:
    """
    Generate what-if scenarios exploring how changing a variable affects the experiment.
    """
    query = (
        f"For the experiment '{topic}' in {course_code}, generate {num_scenarios} detailed "
        f"what-if scenarios exploring the effect of changing {variable}. "
        f"For each scenario: describe the change, predict the outcome, explain the underlying science, "
        f"and state how the virtual simulation would visually show this change."
    )
    system = (
        "You are a STEM educator specialising in inquiry-based learning. "
        "Generate scientifically accurate what-if scenarios that encourage critical thinking."
    )

    result = rag_generate(
        query=query,
        system_instruction=system,
        extra_context=get_course_context(course_code),
    )

    return {
        "scenarios_text": result["answer"],
        "topic": topic,
        "variable_explored": variable,
        "retrieved_chunks": result["retrieved_chunks"],
    }
