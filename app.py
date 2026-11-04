"""
app.py — Flask REST API backend for the AI Lab Manual & Experiment Generator.

Endpoints:
  POST /api/generate/manual        — Generate a full lab manual
  POST /api/generate/safety        — Generate safety instructions
  POST /api/generate/simulation    — Generate virtual experiment simulation
  POST /api/generate/rubric        — Generate evaluation rubric
  POST /api/generate/complete      — Generate all three (manual + simulation + rubric)
  POST /api/curriculum/ingest      — Ingest a curriculum text document
  GET  /api/curriculum/search      — Search curriculum knowledge base
  GET  /api/health                 — Health check
  GET  /api/config/rubric-types    — List available rubric types
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

# ── Local modules ─────────────────────────────────────────────────────────────
from config import Config
from curriculum_retriever import (
    ingest_text_file,
    search_curriculum,
    seed_curriculum_database,
)
from experiment_simulator import generate_virtual_experiment, generate_what_if_scenarios
from lab_manual_generator import generate_lab_manual, generate_safety_instructions
from rubric_generator import (
    generate_evaluation_rubric,
    generate_grade_conversion_scale,
    get_available_rubric_types,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = Config.FLASK_SECRET_KEY
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── Startup ───────────────────────────────────────────────────────────────────
Config.ensure_directories()


@app.before_request
def _log_request():
    logger.info("→ %s %s", request.method, request.path)


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    missing = Config.validate()
    return jsonify(
        {
            "status": "ok" if not missing else "degraded",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "missing_config": missing,
            "model": Config.GRANITE_LLM_MODEL_ID,
            "embedding_model": Config.GRANITE_EMBEDDING_MODEL_ID,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lab manual generation
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/generate/manual", methods=["POST"])
def api_generate_manual():
    """
    Generate a step-by-step lab manual.

    Request JSON:
      topic (str, required)
      course_code (str, default "GENERAL")
      equipment_available (list[str], optional)
      duration_hours (float, default 3.0)
      student_level (str, default "undergraduate")
    """
    data = request.get_json(force=True) or {}
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Field 'topic' is required."}), 400

    try:
        result = generate_lab_manual(
            topic=topic,
            course_code=data.get("course_code", "GENERAL"),
            equipment_available=data.get("equipment_available"),
            duration_hours=float(data.get("duration_hours", 3.0)),
            student_level=data.get("student_level", "undergraduate"),
        )
        return jsonify({"success": True, "data": result})
    except Exception as exc:
        logger.exception("Manual generation failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Safety instructions
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/generate/safety", methods=["POST"])
def api_generate_safety():
    """
    Generate safety instructions for an experiment.

    Request JSON:
      topic (str, required)
      course_code (str, default "GENERAL")
      chemicals (list[str], optional)
      equipment (list[str], optional)
    """
    data = request.get_json(force=True) or {}
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Field 'topic' is required."}), 400

    try:
        result = generate_safety_instructions(
            topic=topic,
            course_code=data.get("course_code", "GENERAL"),
            chemicals=data.get("chemicals"),
            equipment=data.get("equipment"),
        )
        return jsonify({"success": True, "data": result})
    except Exception as exc:
        logger.exception("Safety generation failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Virtual experiment simulation
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/generate/simulation", methods=["POST"])
def api_generate_simulation():
    """
    Generate a virtual experiment simulation guide.

    Request JSON:
      topic (str, required)
      course_code (str, default "GENERAL")
      subject (str, default "General")
      num_trials (int, default 5)
      include_virtual_data (bool, default true)
    """
    data = request.get_json(force=True) or {}
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Field 'topic' is required."}), 400

    try:
        result = generate_virtual_experiment(
            topic=topic,
            course_code=data.get("course_code", "GENERAL"),
            subject=data.get("subject", "General"),
            num_trials=int(data.get("num_trials", 5)),
            include_virtual_data=data.get("include_virtual_data", True),
        )
        return jsonify({"success": True, "data": result})
    except Exception as exc:
        logger.exception("Simulation generation failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation rubric
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/generate/rubric", methods=["POST"])
def api_generate_rubric():
    """
    Generate an evaluation rubric.

    Request JSON:
      topic (str, required)
      course_code (str, default "GENERAL")
      rubric_type (str, default "lab_report")
      learning_objectives (list[str], optional)
      total_points (int, default 100)
    """
    data = request.get_json(force=True) or {}
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Field 'topic' is required."}), 400

    try:
        result = generate_evaluation_rubric(
            topic=topic,
            course_code=data.get("course_code", "GENERAL"),
            rubric_type=data.get("rubric_type", "lab_report"),
            learning_objectives=data.get("learning_objectives"),
            total_points=int(data.get("total_points", 100)),
        )
        grade_scale = generate_grade_conversion_scale(
            total_points=int(data.get("total_points", 100)),
            grading_system=data.get("grading_system", "letter"),
        )
        result["grade_conversion"] = grade_scale
        return jsonify({"success": True, "data": result})
    except Exception as exc:
        logger.exception("Rubric generation failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Complete package: manual + simulation + rubric in one call
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/generate/complete", methods=["POST"])
def api_generate_complete():
    """
    Generate a complete lab package: manual + virtual simulation + rubric.

    Request JSON:
      topic (str, required)
      course_code (str, default "GENERAL")
      subject (str, default "General")
      equipment_available (list[str], optional)
      duration_hours (float, default 3.0)
      student_level (str, default "undergraduate")
      learning_objectives (list[str], optional)
      rubric_type (str, default "lab_report")
    """
    data = request.get_json(force=True) or {}
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Field 'topic' is required."}), 400

    course_code = data.get("course_code", "GENERAL")
    subject = data.get("subject", "General")

    try:
        manual = generate_lab_manual(
            topic=topic,
            course_code=course_code,
            equipment_available=data.get("equipment_available"),
            duration_hours=float(data.get("duration_hours", 3.0)),
            student_level=data.get("student_level", "undergraduate"),
        )
        simulation = generate_virtual_experiment(
            topic=topic,
            course_code=course_code,
            subject=subject,
        )
        rubric = generate_evaluation_rubric(
            topic=topic,
            course_code=course_code,
            rubric_type=data.get("rubric_type", "lab_report"),
            learning_objectives=data.get("learning_objectives"),
        )
        safety = generate_safety_instructions(
            topic=topic,
            course_code=course_code,
        )

        return jsonify(
            {
                "success": True,
                "data": {
                    "topic": topic,
                    "course_code": course_code,
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "manual": manual,
                    "simulation": simulation,
                    "rubric": rubric,
                    "safety": safety,
                },
            }
        )
    except Exception as exc:
        logger.exception("Complete package generation failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum ingestion
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/curriculum/seed", methods=["POST"])
def api_seed_curriculum():
    """Seed the vector store with built-in STEM curriculum data."""
    try:
        count = seed_curriculum_database()
        return jsonify({"success": True, "chunks_ingested": count})
    except Exception as exc:
        logger.exception("Curriculum seeding failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/curriculum/search", methods=["GET"])
def api_search_curriculum():
    """
    Search the curriculum knowledge base.

    Query params: q (str), top_k (int, default 5)
    """
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "Query parameter 'q' is required."}), 400

    top_k = int(request.args.get("top_k", 5))
    results = search_curriculum(q, top_k=top_k)
    return jsonify({"success": True, "query": q, "results": results})


@app.route("/api/curriculum/ingest", methods=["POST"])
def api_ingest_curriculum():
    """
    Ingest a plain-text curriculum document.

    Request JSON:
      text (str, required)
      course_code (str, default "CUSTOM")
      subject (str, default "General")
      doc_id (str, optional)
    """
    from rag_pipeline import ingest_documents

    data = request.get_json(force=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Field 'text' is required."}), 400

    doc = {
        "id": data.get("doc_id") or f"custom_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "text": text,
        "metadata": {
            "course_code": data.get("course_code", "CUSTOM"),
            "subject": data.get("subject", "General"),
        },
    }
    try:
        count = ingest_documents([doc])
        return jsonify({"success": True, "chunks_ingested": count, "doc_id": doc["id"]})
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Configuration endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/config/rubric-types", methods=["GET"])
def api_rubric_types():
    return jsonify({"rubric_types": get_available_rubric_types()})


# ─────────────────────────────────────────────────────────────────────────────
# Serve frontend
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    from flask import send_from_directory
    static_path = Path(app.static_folder or "static")
    if path and (static_path / path).exists():
        return send_from_directory(str(static_path), path)
    index = static_path / "index.html"
    if index.exists():
        return send_from_directory(str(static_path), "index.html")
    return jsonify({"message": "AI Lab Manual & Experiment Generator API", "version": "1.0.0"}), 200


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting AI Lab Manual Generator on %s:%d", Config.FLASK_HOST, Config.FLASK_PORT)
    app.run(
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        debug=Config.FLASK_DEBUG,
    )
