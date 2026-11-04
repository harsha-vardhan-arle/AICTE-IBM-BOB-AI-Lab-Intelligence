"""
curriculum_retriever.py — Curriculum data retrieval & ingestion module.

Handles:
  - Loading syllabus documents from local corpus directory
  - Parsing PDF and plain-text curriculum files
  - Seeding the vector store with built-in STEM sample curriculum data
  - Querying by course code, topic, or subject area
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from config import Config
from rag_pipeline import ingest_documents, retrieve_context

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Built-in STEM curriculum seed data
# ─────────────────────────────────────────────────────────────────────────────

SEED_CURRICULUM: list[dict[str, Any]] = [
    {
        "id": "PHY101_mechanics",
        "text": (
            "Course: PHY101 – Classical Mechanics. Topics: Newton's Laws of Motion, "
            "Work and Energy, Momentum and Collisions, Rotational Dynamics, Simple Harmonic Motion, "
            "Gravitation. Lab Equipment: air track, force sensors, spring scales, ticker-tape timer, "
            "pendulum apparatus, inclined planes. Safety: eye protection required; secure all glassware."
        ),
        "metadata": {"course_code": "PHY101", "subject": "Physics", "level": "undergraduate"},
    },
    {
        "id": "CHM101_general_chem",
        "text": (
            "Course: CHM101 – General Chemistry. Topics: Stoichiometry, Acid-Base Reactions, "
            "Redox Chemistry, Thermochemistry, Chemical Kinetics, Electrochemistry. "
            "Lab Equipment: burettes, pipettes, beakers, Bunsen burners, pH meters, spectrophotometer. "
            "Safety: lab coats and gloves mandatory; fume hood for volatile reagents; "
            "neutralise acid spills with sodium bicarbonate."
        ),
        "metadata": {"course_code": "CHM101", "subject": "Chemistry", "level": "undergraduate"},
    },
    {
        "id": "BIO101_cell_biology",
        "text": (
            "Course: BIO101 – Cell Biology. Topics: Cell Structure, Membrane Transport, "
            "Cellular Respiration, Photosynthesis, Mitosis and Meiosis, DNA Replication. "
            "Lab Equipment: compound microscope, centrifuge, gel electrophoresis apparatus, "
            "micropipettes, hemocytometer. "
            "Safety: biohazard waste disposal; autoclave contaminated materials; "
            "handle formaldehyde in fume hood."
        ),
        "metadata": {"course_code": "BIO101", "subject": "Biology", "level": "undergraduate"},
    },
    {
        "id": "CS201_data_structures",
        "text": (
            "Course: CS201 – Data Structures & Algorithms. Topics: Arrays and Linked Lists, "
            "Stacks and Queues, Trees and Graphs, Sorting Algorithms (QuickSort, MergeSort, HeapSort), "
            "Hashing, Dynamic Programming. Lab Equipment: computer workstations with Python/Java IDEs, "
            "whiteboard for algorithm design. "
            "Assessment: code correctness, time complexity analysis, unit test coverage."
        ),
        "metadata": {"course_code": "CS201", "subject": "Computer Science", "level": "undergraduate"},
    },
    {
        "id": "ECE301_circuits",
        "text": (
            "Course: ECE301 – Electric Circuits. Topics: Ohm's Law, Kirchhoff's Laws, "
            "AC/DC Circuit Analysis, Operational Amplifiers, RC/RL/RLC Circuits, Frequency Response. "
            "Lab Equipment: digital multimeter, oscilloscope, function generator, breadboards, "
            "resistors/capacitors/inductors kit. "
            "Safety: never exceed 50V without supervision; discharge capacitors before handling."
        ),
        "metadata": {"course_code": "ECE301", "subject": "Electrical Engineering", "level": "undergraduate"},
    },
    {
        "id": "ENV201_environmental_science",
        "text": (
            "Course: ENV201 – Environmental Science. Topics: Ecosystem Dynamics, "
            "Water Quality Analysis, Air Pollution Monitoring, Soil Composition, "
            "Climate Change Indicators, Biodiversity Sampling. "
            "Lab Equipment: water testing kits, spectrophotometer, soil auger, pH meters, "
            "turbidity meters. "
            "Safety: field safety briefing required; avoid chemical runoff into water sources."
        ),
        "metadata": {"course_code": "ENV201", "subject": "Environmental Science", "level": "undergraduate"},
    },
    {
        "id": "MATH301_numerical_methods",
        "text": (
            "Course: MATH301 – Numerical Methods. Topics: Root Finding (Bisection, Newton-Raphson), "
            "Numerical Integration (Simpson's Rule, Gaussian Quadrature), "
            "Solving ODEs (Euler, Runge-Kutta), Matrix Methods, Interpolation. "
            "Lab Equipment: MATLAB/Python computing environment. "
            "Assessment: accuracy of numerical approximations, convergence analysis, error bounds."
        ),
        "metadata": {"course_code": "MATH301", "subject": "Mathematics", "level": "undergraduate"},
    },
    {
        "id": "safety_general",
        "text": (
            "General Laboratory Safety Guidelines: "
            "1. Always wear personal protective equipment (PPE): lab coat, safety goggles, gloves. "
            "2. Know the location of fire extinguishers, eye wash stations, and emergency exits. "
            "3. Never eat, drink, or apply cosmetics in the laboratory. "
            "4. Label all chemical containers clearly with name, concentration, and date. "
            "5. Dispose of chemical waste according to institutional hazardous waste protocols. "
            "6. Report all accidents, spills, or near-misses to the supervising instructor immediately. "
            "7. Tie back long hair and avoid loose clothing near open flames or rotating equipment. "
            "8. Wash hands thoroughly before leaving the laboratory."
        ),
        "metadata": {"course_code": "GENERAL", "subject": "Safety", "level": "all"},
    },
    {
        "id": "assessment_rubric_general",
        "text": (
            "Standard Lab Assessment Rubric (100 points total): "
            "Pre-Lab Preparation (15 pts): hypothesis stated, materials listed, procedure outlined. "
            "Safety Compliance (10 pts): PPE worn, waste disposed correctly, no violations. "
            "Experimental Procedure (25 pts): accurate execution, proper technique, reproducibility. "
            "Data Collection (20 pts): organised data tables, correct units, significant figures. "
            "Analysis & Calculations (20 pts): correct formulas, error analysis, graphing. "
            "Conclusion (10 pts): hypothesis evaluated, sources of error identified, future improvements."
        ),
        "metadata": {"course_code": "GENERAL", "subject": "Assessment", "level": "all"},
    },
    {
        "id": "remote_lab_guidelines",
        "text": (
            "Remote & Virtual Lab Guidelines: "
            "Virtual simulations approved: PhET Interactive Simulations (physics/chemistry), "
            "Labster (biology/chemistry), ChemCollective, CircuitLab (electronics). "
            "Data collection from virtual labs must include screenshots of results. "
            "Uncertainty analysis applies equally to virtual experiments. "
            "Students must document each step as if performing the physical experiment. "
            "Submission: lab report as PDF, simulation screenshots attached, data in Excel/CSV."
        ),
        "metadata": {"course_code": "GENERAL", "subject": "Remote Labs", "level": "all"},
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def seed_curriculum_database() -> int:
    """
    Ingest the built-in seed curriculum into the vector store.
    Returns the number of chunks stored.
    """
    logger.info("Seeding vector store with built-in curriculum data (%d documents)…", len(SEED_CURRICULUM))
    return ingest_documents(SEED_CURRICULUM)


def ingest_text_file(file_path: str, course_code: str = "CUSTOM", subject: str = "General") -> int:
    """Ingest a plain-text curriculum file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    text = path.read_text(encoding="utf-8")
    doc = {
        "id": f"file__{path.stem}",
        "text": text,
        "metadata": {"course_code": course_code, "subject": subject, "source_file": path.name},
    }
    return ingest_documents([doc])


def ingest_pdf_file(file_path: str, course_code: str = "CUSTOM", subject: str = "General") -> int:
    """Ingest a PDF curriculum file (requires PyPDF2)."""
    try:
        import PyPDF2  # type: ignore
    except ImportError:
        raise ImportError("PyPDF2 is required for PDF ingestion. Install with: pip install PyPDF2")

    path = Path(file_path)
    pages: list[str] = []
    with open(path, "rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        for page in reader.pages:
            pages.append(page.extract_text() or "")

    full_text = "\n".join(pages)
    doc = {
        "id": f"pdf__{path.stem}",
        "text": full_text,
        "metadata": {"course_code": course_code, "subject": subject, "source_file": path.name},
    }
    return ingest_documents([doc])


def search_curriculum(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Search the curriculum vector store for chunks relevant to *query*.
    Returns a list of result dicts.
    """
    return retrieve_context(query, top_k=top_k)


def get_course_context(course_code: str) -> str:
    """
    Return all seed curriculum text for a specific course code as a single string.
    Used to inject precise course context into generation prompts.
    """
    matches = [
        doc["text"]
        for doc in SEED_CURRICULUM
        if doc["metadata"].get("course_code", "").upper() == course_code.upper()
    ]
    return "\n\n".join(matches)
