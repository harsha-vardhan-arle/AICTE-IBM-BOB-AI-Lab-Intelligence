# 🧪 AI Lab Manual & Experiment Generator

> **RAG-powered STEM laboratory assistant built on IBM Granite + watsonx.ai**

An AI-driven assistant that helps educators and students generate step-by-step lab manuals, virtual experiment simulations, and evaluation rubrics — all tailored to their course syllabus and available equipment.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📖 **Lab Manual Generator** | Complete manuals with objectives, theory, procedure, data tables, and discussion questions |
| 🔬 **Virtual Experiment Simulator** | Step-by-step simulation guides for PhET, Labster, CircuitLab, and more |
| 📊 **Evaluation Rubric Generator** | Bloom's Taxonomy-aligned rubrics with grade conversion scales |
| ⚠️ **Safety Instruction Generator** | PPE, chemical hazards, first aid, and emergency protocols |
| 🚀 **Complete Package** | All four deliverables generated in one request |
| 📚 **Curriculum Knowledge Base** | ChromaDB vector store seeded with 7 STEM courses |
| 🔍 **Semantic Search** | RAG retrieval using IBM Granite embeddings |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (HTML/JS)                    │
│   Dashboard · Manual · Simulation · Rubric · Safety      │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP REST API
┌────────────────────────▼────────────────────────────────┐
│              Flask Backend (app.py)                      │
│  /api/generate/manual  /api/generate/simulation          │
│  /api/generate/rubric  /api/generate/safety              │
│  /api/generate/complete  /api/curriculum/*               │
└──┬────────────────┬────────────────────┬────────────────┘
   │                │                    │
   ▼                ▼                    ▼
lab_manual_    experiment_         rubric_
generator.py   simulator.py        generator.py
   │                │                    │
   └────────────────┴────────────────────┘
                    │
         ┌──────────▼──────────┐
         │   rag_pipeline.py   │
         │  (Core RAG Engine)  │
         └──────┬──────────────┘
                │
     ┌──────────┼──────────────┐
     ▼          ▼              ▼
 ChromaDB   IBM Granite    IBM Granite
 Vector     Embeddings     LLM Inference
 Store      (watsonx.ai)   (watsonx.ai)
```

### Technology Stack

| Layer | Technology |
|---|---|
| **LLM** | IBM Granite 4.0 Small (`ibm/granite-4-h-small`) |
| **Embeddings** | IBM Granite Embedding 278M (`ibm/granite-embedding-278m-multilingual`) |
| **AI Platform** | IBM watsonx.ai (IBM Cloud Lite) |
| **Vector Store** | ChromaDB (persistent, local) |
| **RAG Framework** | LangChain + custom pipeline |
| **Backend** | Flask 3.x + Flask-CORS |
| **Frontend** | Vanilla HTML/CSS/JS (IBM Carbon Design) |

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/harsha-vardhan-arle/AICTE-IBM-BOB-AI-Lab-Intelligence.git
cd AICTE-IBM-BOB-AI-Lab-Intelligence
pip install -r requirements.txt
```

### 2. Configure credentials

Copy `env.txt` to `.env` in the project root and fill in your IBM Cloud credentials:

```bash
cp env.txt .env
```

Edit `.env`:
```ini
IBM_CLOUD_API_KEY=your_actual_api_key
IBM_WATSONX_PROJECT_ID=your_actual_project_id
IBM_CLOUD_URL=https://us-south.ml.cloud.ibm.com
```

> **Getting IBM Cloud credentials (Free Lite tier):**
> 1. Sign up at [cloud.ibm.com](https://cloud.ibm.com) (free, no credit card)
> 2. Create a **watsonx.ai Studio** service (Lite plan)
> 3. Go to **Manage → Access (IAM) → API Keys** → Create API key
> 4. In watsonx.ai Studio, copy your **Project ID** from the project settings

### 3. Start the server

```bash
python app.py
```

Open your browser at **http://localhost:5000**

### 4. Seed the curriculum database

Click **📚 Curriculum Manager → 🌱 Seed Curriculum Database** in the UI, or run:

```bash
curl -X POST http://localhost:5000/api/curriculum/seed
```

---

## 📡 REST API Reference

### Health Check
```
GET /api/health
```

### Generate Lab Manual
```
POST /api/generate/manual
Content-Type: application/json

{
  "topic": "Acid-Base Titration",
  "course_code": "CHM101",
  "equipment_available": ["burette", "pipette", "pH meter"],
  "duration_hours": 3.0,
  "student_level": "undergraduate"
}
```

### Generate Virtual Simulation
```
POST /api/generate/simulation
{
  "topic": "Simple Pendulum",
  "course_code": "PHY101",
  "subject": "physics",
  "num_trials": 5,
  "include_virtual_data": true
}
```

### Generate Evaluation Rubric
```
POST /api/generate/rubric
{
  "topic": "Ohm's Law Verification",
  "course_code": "ECE301",
  "rubric_type": "lab_report",
  "learning_objectives": ["Apply Kirchhoff's Laws", "Analyse V-I characteristics"],
  "total_points": 100
}
```
> Rubric types: `lab_report` | `virtual_lab` | `practical_exam` | `group_project`

### Generate Safety Instructions
```
POST /api/generate/safety
{
  "topic": "Flame Test for Metal Ions",
  "course_code": "CHM101",
  "chemicals": ["HCl", "metal salt solutions"],
  "equipment": ["Bunsen burner", "nichrome wire", "safety goggles"]
}
```

### Generate Complete Package
```
POST /api/generate/complete
{
  "topic": "Acid-Base Titration",
  "course_code": "CHM101",
  "subject": "chemistry",
  "duration_hours": 3.0,
  "student_level": "undergraduate",
  "rubric_type": "lab_report"
}
```

### Seed Curriculum
```
POST /api/curriculum/seed
```

### Search Curriculum
```
GET /api/curriculum/search?q=Newton%27s+laws&top_k=5
```

### Ingest Custom Document
```
POST /api/curriculum/ingest
{
  "text": "Course: CHM301 – Organic Chemistry. Topics: ...",
  "course_code": "CHM301",
  "subject": "Organic Chemistry"
}
```

---

## 📚 Built-in Curriculum Coverage

| Course Code | Subject | Topics |
|---|---|---|
| PHY101 | Classical Mechanics | Newton's Laws, Work-Energy, SHM, Gravitation |
| CHM101 | General Chemistry | Stoichiometry, Acid-Base, Redox, Kinetics |
| BIO101 | Cell Biology | Cell Structure, Respiration, Photosynthesis, Mitosis |
| CS201 | Computer Science | Data Structures, Sorting, Graphs, DP |
| ECE301 | Electrical Engineering | Ohm's Law, Kirchhoff's Laws, Op-Amps, RLC |
| ENV201 | Environmental Science | Water Quality, Air Pollution, Soil Analysis |
| MATH301 | Numerical Methods | Root Finding, Integration, ODE Solvers |
| GENERAL | Safety | PPE, Hazardous Waste, Emergency Protocols |
| GENERAL | Assessment | Standard 100-point lab rubric |
| GENERAL | Remote Labs | PhET, Labster, ChemCollective, CircuitLab |

---

## 🗂️ Project Structure

```
ai-lab-manual-generator/
├── app.py                    # Flask REST API (10 endpoints)
├── config.py                 # Environment config loader
├── rag_pipeline.py           # Core RAG engine (IBM Granite + ChromaDB)
├── curriculum_retriever.py   # Curriculum ingestion & retrieval
├── lab_manual_generator.py   # Lab manual generation
├── experiment_simulator.py   # Virtual simulation generation
├── rubric_generator.py       # Rubric & grade scale generation
├── requirements.txt          # Python dependencies
├── env.txt                   # Environment template (copy to .env)
├── static/
│   └── index.html            # Full-featured frontend UI
└── data/
    ├── chroma_db/            # ChromaDB vector store (auto-created)
    ├── corpus/               # Local curriculum files
    ├── output/               # Generated manuals (PDF export)
    └── uploads/              # User-uploaded syllabus files
```

---

## 🔒 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `IBM_CLOUD_API_KEY` | ✅ | IBM Cloud IAM API key |
| `IBM_WATSONX_PROJECT_ID` | ✅ | watsonx.ai Studio project ID |
| `IBM_CLOUD_URL` | ✅ | watsonx.ai endpoint URL |
| `GRANITE_LLM_MODEL_ID` | — | LLM model (default: `ibm/granite-4-h-small`) |
| `GRANITE_EMBEDDING_MODEL_ID` | — | Embedding model (default: `ibm/granite-embedding-278m-multilingual`) |
| `RAG_CHUNK_SIZE` | — | Token chunk size for ingestion (default: 512) |
| `RAG_TOP_K_RESULTS` | — | Retrieval top-k (default: 5) |
| `LLM_MAX_NEW_TOKENS` | — | Max generation tokens (default: 1024) |
| `FLASK_PORT` | — | Server port (default: 5000) |

---

## 📄 License

MIT License — free for educational and commercial use.

---

*Built with IBM Granite · IBM watsonx.ai · ChromaDB · Flask*
