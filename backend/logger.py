"""
logger.py — Dual-Sink Incident Logger (SQLite + JSON Lines)
=============================================================

Architecture:
    DetectionResult → IncidentLogger → [SQLite DB, JSONL file]
    
    Both sinks are written atomically. If one fails, the other still records.
    This creates the "real-world impact" dataset referenced in the research paper.

Design Decisions:
    1. Dual sinks: SQLite for structured queries (dashboards, analytics),
       JSONL for append-only streaming (log shipping, Kafka ingest, paper evidence).
    2. Thread-safe: Uses threading.Lock for SQLite writes (sqlite3 is NOT
       safe for concurrent writes without WAL mode + mutex).
    3. WAL mode: Enables concurrent readers while writing. Critical for the
       FastAPI endpoint that reads incidents while the inference loop writes.
    4. Deduplication: Uses a cooldown window per person_id to prevent
       logging the same ongoing event 15 times per second.

Author: Violence Detection System
"""

import json
import sqlite3
import threading
import time
import uuid
import logging
import os
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
from dataclasses import asdict

from inference import DetectionResult, Severity

logger = logging.getLogger("incident_logger")

# Default data directory
DATA_DIR = Path(__file__).parent / "data"


class IncidentLogger:
    """
    Thread-safe dual-sink logger that records violence detection incidents
    to both SQLite and JSON Lines format.

    Features:
        - Atomic writes to both sinks
        - Per-person cooldown to prevent duplicate logging
        - WAL mode for concurrent read/write
        - Auto-creates data directory and schema
        - Queryable incident history via get_recent_incidents()
    """

    # SQLite schema
    _CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS incidents (
        id TEXT PRIMARY KEY,
        timestamp REAL NOT NULL,
        timestamp_iso TEXT NOT NULL,
        person_id INTEGER NOT NULL,
        severity INTEGER NOT NULL,
        severity_label TEXT NOT NULL,
        max_velocity REAL NOT NULL,
        mean_velocity REAL NOT NULL,
        confidence REAL NOT NULL,
        bbox_x1 REAL NOT NULL,
        bbox_y1 REAL NOT NULL,
        bbox_x2 REAL NOT NULL,
        bbox_y2 REAL NOT NULL,
        frame_width INTEGER DEFAULT 640,
        frame_height INTEGER DEFAULT 480
    );
    """

    _CREATE_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_incidents_timestamp 
    ON incidents(timestamp DESC);
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        jsonl_path: Optional[str] = None,
        cooldown_seconds: float = 2.0,
        data_dir: Optional[str] = None,
    ):
        """
        Args:
            db_path: Path to SQLite database. Defaults to data/incidents.db
            jsonl_path: Path to JSONL log file. Defaults to data/incidents.jsonl
            cooldown_seconds: Minimum seconds between logs for the same person_id
            data_dir: Base data directory. Defaults to ./data/
        """
        self._data_dir = Path(data_dir) if data_dir else DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = db_path or str(self._data_dir / "incidents.db")
        self._jsonl_path = jsonl_path or str(self._data_dir / "incidents.jsonl")
        self._cooldown = cooldown_seconds

        # Per-person last-log timestamp for deduplication
        self._last_logged: dict[int, float] = {}

        # Thread safety
        self._lock = threading.Lock()

        # Initialize SQLite
        self._init_db()

        logger.info(
            f"IncidentLogger initialized — "
            f"DB: {self._db_path}, JSONL: {self._jsonl_path}, "
            f"Cooldown: {self._cooldown}s"
        )

    def _init_db(self):
        """Create database, tables, and enable WAL mode."""
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(self._CREATE_TABLE)
            conn.execute(self._CREATE_INDEX)
            conn.commit()
            logger.info(f"SQLite database initialized: {self._db_path}")
        finally:
            conn.close()

    def _should_log(self, person_id: int, now: float) -> bool:
        """Check cooldown window for deduplication."""
        last = self._last_logged.get(person_id, 0.0)
        return (now - last) >= self._cooldown

    def _build_record(self, detection: DetectionResult) -> dict:
        """Convert DetectionResult to a serializable dictionary."""
        now = detection.timestamp
        iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
        
        return {
            "id": str(uuid.uuid4()),
            "timestamp": now,
            "timestamp_iso": iso,
            "person_id": detection.person_id,
            "severity": detection.severity.value,
            "severity_label": detection.severity.name,
            "max_velocity": round(detection.max_velocity, 2),
            "mean_velocity": round(detection.mean_velocity, 2),
            "confidence": round(detection.confidence, 4),
            "bbox_x1": round(detection.bbox[0], 1),
            "bbox_y1": round(detection.bbox[1], 1),
            "bbox_x2": round(detection.bbox[2], 1),
            "bbox_y2": round(detection.bbox[3], 1),
        }

    def _write_sqlite(self, record: dict):
        """Write a single record to SQLite."""
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                """INSERT INTO incidents 
                   (id, timestamp, timestamp_iso, person_id, severity, 
                    severity_label, max_velocity, mean_velocity, confidence,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record["id"],
                    record["timestamp"],
                    record["timestamp_iso"],
                    record["person_id"],
                    record["severity"],
                    record["severity_label"],
                    record["max_velocity"],
                    record["mean_velocity"],
                    record["confidence"],
                    record["bbox_x1"],
                    record["bbox_y1"],
                    record["bbox_x2"],
                    record["bbox_y2"],
                )
            )
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"SQLite write failed: {e}")
        finally:
            conn.close()

    def _write_jsonl(self, record: dict):
        """Append a single record to JSONL file."""
        try:
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except IOError as e:
            logger.error(f"JSONL write failed: {e}")

    def log_detection(self, detection: DetectionResult) -> Optional[dict]:
        """
        Log a violence detection event if it passes the cooldown filter.

        Args:
            detection: DetectionResult from the inference engine.

        Returns:
            The logged record dict, or None if filtered by cooldown.
        """
        now = detection.timestamp

        with self._lock:
            # Cooldown deduplication
            if not self._should_log(detection.person_id, now):
                return None

            # Build record
            record = self._build_record(detection)

            # Write to both sinks (independent — one failure doesn't block the other)
            self._write_sqlite(record)
            self._write_jsonl(record)

            # Update cooldown tracker
            self._last_logged[detection.person_id] = now

            logger.info(
                f"Incident logged: {record['severity_label']} "
                f"(person={record['person_id']}, "
                f"vel={record['max_velocity']}px/f)"
            )

            return record

    def log_detections(self, detections: list[DetectionResult]) -> list[dict]:
        """Batch log multiple detections. Returns list of successfully logged records."""
        records = []
        for det in detections:
            record = self.log_detection(det)
            if record is not None:
                records.append(record)
        return records

    def get_recent_incidents(
        self,
        limit: int = 50,
        min_severity: int = 1,
        since_timestamp: Optional[float] = None,
    ) -> list[dict]:
        """
        Query recent incidents from SQLite.
        Used by the FastAPI endpoint and the frontend.

        Args:
            limit: Maximum number of incidents to return.
            min_severity: Minimum severity level (1=LOW, 2=MEDIUM, 3=HIGH, 4=CRITICAL).
            since_timestamp: Only return incidents after this Unix timestamp.

        Returns:
            List of incident dicts, ordered by timestamp descending.
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            query = """
                SELECT * FROM incidents 
                WHERE severity >= ?
            """
            params: list = [min_severity]

            if since_timestamp is not None:
                query += " AND timestamp > ?"
                params.append(since_timestamp)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"SQLite query failed: {e}")
            return []
        finally:
            conn.close()

    def get_incident_count(self, since_hours: float = 24.0) -> dict:
        """
        Get incident counts grouped by severity for the last N hours.
        Used for dashboard statistics.
        """
        cutoff = time.time() - (since_hours * 3600)
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(
                """SELECT severity_label, COUNT(*) as count 
                   FROM incidents 
                   WHERE timestamp > ? 
                   GROUP BY severity_label""",
                (cutoff,)
            )
            counts = {row[0]: row[1] for row in cursor.fetchall()}
            return {
                "LOW": counts.get("LOW", 0),
                "MEDIUM": counts.get("MEDIUM", 0),
                "HIGH": counts.get("HIGH", 0),
                "CRITICAL": counts.get("CRITICAL", 0),
                "total": sum(counts.values()),
                "since_hours": since_hours,
            }
        except sqlite3.Error as e:
            logger.error(f"SQLite count query failed: {e}")
            return {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0, "total": 0}
        finally:
            conn.close()

    def clear_all(self):
        """Purge all logged data. Use for testing only."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute("DELETE FROM incidents")
                conn.commit()
            finally:
                conn.close()

            # Truncate JSONL
            open(self._jsonl_path, "w").close()
            self._last_logged.clear()
            logger.warning("All incident data purged.")
