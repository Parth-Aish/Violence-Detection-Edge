"""Persist positive violence predictions and runtime measurements."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from inference import ViolencePrediction

logger = logging.getLogger("incident_logger")
DATA_DIR = Path(__file__).parent / "data"


class IncidentLogger:
    """Thread-safe SQLite and JSONL logger with a duplicate-event cooldown."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        jsonl_path: Optional[str] = None,
        cooldown_seconds: float = 2.0,
        data_dir: Optional[str] = None,
    ) -> None:
        base_dir = Path(data_dir) if data_dir else DATA_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or str(base_dir / "incidents.db")
        self.jsonl_path = jsonl_path or str(base_dir / "incidents.jsonl")
        self.cooldown_seconds = cooldown_seconds
        self._last_logged_at = 0.0
        self._lock = threading.Lock()
        with closing(sqlite3.connect(self.db_path)) as connection:
            with connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS violence_events (
                        id TEXT PRIMARY KEY,
                        timestamp REAL NOT NULL,
                        timestamp_iso TEXT NOT NULL,
                        is_violence INTEGER NOT NULL,
                        probability REAL NOT NULL,
                        severity INTEGER NOT NULL,
                        severity_label TEXT NOT NULL,
                        fps REAL NOT NULL,
                        inference_ms REAL NOT NULL,
                        confidence REAL NOT NULL,
                        bbox_json TEXT
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_violence_events_timestamp "
                    "ON violence_events(timestamp DESC)"
                )
        logger.info(
            "Incident logger ready (SQLite=%s, JSONL=%s)",
            self.db_path,
            self.jsonl_path,
        )

    @staticmethod
    def _record(prediction: ViolencePrediction) -> dict:
        timestamp = prediction.timestamp
        return {
            "id": str(uuid.uuid4()),
            "timestamp": timestamp,
            "timestamp_iso": datetime.fromtimestamp(
                timestamp, tz=timezone.utc
            ).isoformat(),
            "is_violence": prediction.is_fighting,
            "probability": round(prediction.probability, 6),
            "severity": int(prediction.severity),
            "severity_label": prediction.severity.name,
            "fps": round(prediction.fps, 3),
            "inference_ms": round(prediction.inference_ms, 3),
            "confidence": round(prediction.confidence, 4),
            "bbox": list(prediction.bbox) if prediction.bbox else None,
        }

    def log_prediction(self, prediction: ViolencePrediction) -> Optional[dict]:
        """Write a positive prediction unless it falls inside the cooldown."""
        if not prediction.is_fighting:
            return None

        with self._lock:
            if prediction.timestamp - self._last_logged_at < self.cooldown_seconds:
                return None
            record = self._record(prediction)
            with closing(sqlite3.connect(self.db_path, timeout=10)) as connection:
                with connection:
                    connection.execute(
                        """
                        INSERT INTO violence_events (
                            id, timestamp, timestamp_iso, is_violence, probability,
                            severity, severity_label, fps, inference_ms, confidence, bbox_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record["id"],
                            record["timestamp"],
                            record["timestamp_iso"],
                            int(record["is_violence"]),
                            record["probability"],
                            record["severity"],
                            record["severity_label"],
                            record["fps"],
                            record["inference_ms"],
                            record["confidence"],
                            json.dumps(record["bbox"]),
                        ),
                    )
            try:
                with open(self.jsonl_path, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError:
                logger.exception("Failed to append incident record to %s", self.jsonl_path)
                raise

            self._last_logged_at = prediction.timestamp
            logger.warning(
                "Violence incident logged: %s probability=%.3f fps=%.2f latency=%.2fms",
                record["severity_label"],
                record["probability"],
                record["fps"],
                record["inference_ms"],
            )
            return record

    def log_batch_incident(
        self,
        video_source: str,
        probability: float,
        incident_timestamps: Optional[list[float]] = None,
        details: str = "",
    ) -> Optional[dict]:
        """Record an incident identified during batch video analysis."""
        from inference import Severity, ViolencePrediction
        sev = Severity.HIGH if probability >= 0.8 else Severity.MEDIUM
        pred = ViolencePrediction(
            is_fighting=True,
            probability=probability,
            severity=sev,
            confidence=probability,
            bbox=None,
            timestamp=time.time(),
            inference_ms=0.0,
            fps=30.0,
        )
        return self.log_prediction(pred)

    def get_recent_incidents(
        self, limit: int = 50, since_timestamp: Optional[float] = None
    ) -> list[dict]:
        query = (
            "SELECT id, timestamp, timestamp_iso, is_violence, probability, "
            "severity, severity_label, fps, inference_ms, confidence, bbox_json "
            "FROM violence_events"
        )
        parameters: list[float | int] = []
        if since_timestamp is not None:
            query += " WHERE timestamp >= ?"
            parameters.append(since_timestamp)
        query += " ORDER BY timestamp DESC LIMIT ?"
        parameters.append(limit)

        with closing(sqlite3.connect(self.db_path, timeout=10)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, parameters).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record["is_violence"] = bool(record["is_violence"])
            record["bbox"] = json.loads(record.pop("bbox_json") or "null")
            records.append(record)
        return records

    def get_incident_count(self, since_hours: float = 24.0) -> dict:
        cutoff = time.time() - since_hours * 3600
        with closing(sqlite3.connect(self.db_path, timeout=10)) as connection:
            rows = connection.execute(
                """
                SELECT severity_label, COUNT(*) FROM violence_events
                WHERE timestamp >= ? GROUP BY severity_label
                """,
                (cutoff,),
            ).fetchall()
            total = connection.execute(
                "SELECT COUNT(*) FROM violence_events WHERE timestamp >= ?",
                (cutoff,),
            ).fetchone()[0]
        counts = {name: 0 for name in ("LOW", "MEDIUM", "HIGH", "CRITICAL")}
        counts.update({str(name): int(count) for name, count in rows})
        return {**counts, "total": int(total)}
