# Violence Detection System — Build Progress

> **Project**: Edge-Optimized Spatio-Temporal Violence Detection for Legacy Surveillance Infrastructure  
> **Author**: L6 SWE / AI Researcher  
> **Last Updated**: 2026-10-22T21:42:00+05:30

---

## Backend Core (Python / FastAPI)

- [x] `backend/ingest.py` — Producer-Consumer RTSP ingestion via FFmpeg + multiprocessing.Queue
- [x] `backend/inference.py` — YOLOv8-Pose extraction + temporal velocity heuristic classifier
- [x] `backend/logger.py` — JSON + SQLite incident logger (timestamps, bboxes, severity)
- [x] `backend/main.py` — FastAPI server, process orchestration, WebSocket `/ws/stream`
- [x] `backend/requirements.txt` — Python dependency manifest

## Real-Time Logging & Data Pipeline

- [x] `backend/data/incidents.db` — SQLite schema auto-created at runtime
- [x] `backend/data/incidents.jsonl` — Append-only JSON Lines log (created at runtime)

## Frontend UI (Next.js + Tailwind CSS)

- [x] `frontend/` — Next.js project scaffold
- [x] `frontend/src/app/page.tsx` — Main UI: dark mode, glassmorphism, live feed, incident log
- [x] `frontend/src/app/layout.tsx` — Root layout with metadata and fonts
- [x] `frontend/src/app/globals.css` — Tailwind + custom glassmorphism utilities
- [x] `frontend/tailwind.config.ts` — Using Tailwind v4 (config-free, CSS-first)

## Research Paper

- [x] `paper/paper.md` — IEEE-formatted research paper (full sections)
- [x] `paper/figures/` — Architecture diagrams (described in markdown/mermaid)

## Configuration

- [x] `.gitignore` — Project-level gitignore
- [x] `README.md` — Setup, usage, and deployment instructions

---

## Status Legend

| Symbol | Meaning        |
|--------|----------------|
| `[ ]`  | Not started    |
| `[~]`  | In progress    |
| `[x]`  | Complete       |
| `[!]`  | Blocked        |
