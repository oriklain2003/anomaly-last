"""
Utility script to project feedback flights into a new, query-friendly database.

Steps:
1) Read every row from training_ops/feedback.db (user_feedback table).
2) For each flight_id, fetch the full track from training/research DBs.
3) Fetch the latest anomaly_report for that flight (research/live DBs).
4) Write everything into present_anomalies.db with:
   - anomaly_reports: one row per feedback item, flattened rule metadata.
   - rule_matches: one row per evaluated rule (opens the rules JSON).
   - flight_tracks: all track points with the user label attached.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
FEEDBACK_DB = ROOT / "training_ops" / "feedback.db"
TRAINING_DB = ROOT / "training_ops" / "training_dataset.db"
RESEARCH_DB = ROOT / "realtime" / "research.db"
LIVE_ANOMALIES_DB = ROOT / "realtime" / "live_anomalies.db"
LAST_DB = ROOT / "last.db"  # Fallback for tracks if needed

PRESENT_DB = ROOT / "present_anomalies.db"


@dataclass
class FeedbackRow:
    id: int
    flight_id: str
    timestamp: int
    user_label: int
    comments: str
    model_version: Optional[str]
    rule_id: Optional[int]
    other_details: Optional[str]
    full_report_json: Optional[str]


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def load_feedback_rows() -> List[FeedbackRow]:
    if not FEEDBACK_DB.exists():
        raise FileNotFoundError(f"Feedback DB not found at {FEEDBACK_DB}")

    # Check for full_report_json column
    with connect(FEEDBACK_DB) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(user_feedback)")
        cols = [r["name"] for r in cursor.fetchall()]
        has_report_col = "full_report_json" in cols

        query = """
            SELECT id, flight_id, timestamp, user_label, comments, model_version, rule_id, other_details
            """
        if has_report_col:
            query += ", full_report_json"
        
        query += """
            FROM user_feedback
            ORDER BY id ASC
            """
        
        rows = conn.execute(query).fetchall()

    return [
        FeedbackRow(
            id=row["id"],
            flight_id=row["flight_id"],
            timestamp=row["timestamp"],
            user_label=row["user_label"],
            comments=row["comments"],
            model_version=row["model_version"],
            rule_id=row["rule_id"],
            other_details=row["other_details"],
            full_report_json=row["full_report_json"] if has_report_col else None,
        )
        for row in rows
    ]


def ensure_present_schema() -> None:
    PRESENT_DB.parent.mkdir(parents=True, exist_ok=True)
    with connect(PRESENT_DB) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS anomaly_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flight_id TEXT NOT NULL,
                feedback_id INTEGER,
                feedback_timestamp INTEGER,
                user_label INTEGER,
                comments TEXT,
                model_version TEXT,
                rule_id INTEGER,
                other_details TEXT,
                anomaly_timestamp INTEGER,
                pipeline_is_anomaly INTEGER,
                severity_cnn REAL,
                severity_dense REAL,
                confidence_score REAL,
                summary_triggers TEXT,
                rules_status TEXT,
                rules_triggers TEXT,
                matched_rule_ids TEXT,
                matched_rule_names TEXT,
                matched_rule_categories TEXT,
                full_report_json TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rule_matches (
                report_id INTEGER NOT NULL,
                rule_id INTEGER,
                rule_name TEXT,
                category TEXT,
                severity REAL,
                matched INTEGER,
                summary TEXT,
                details TEXT,
                FOREIGN KEY(report_id) REFERENCES anomaly_reports(id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS flight_tracks (
                flight_id TEXT NOT NULL,
                timestamp INTEGER,
                lat REAL,
                lon REAL,
                alt REAL,
                heading REAL,
                gspeed REAL,
                vspeed REAL,
                track REAL,
                squawk TEXT,
                callsign TEXT,
                source TEXT,
                user_label INTEGER,
                UNIQUE(flight_id, timestamp)
            )
            """
        )

        cur.execute("CREATE INDEX IF NOT EXISTS idx_reports_fid ON anomaly_reports(flight_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tracks_fid_ts ON flight_tracks(flight_id, timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rules_report ON rule_matches(report_id)")
        conn.commit()


def _query_track_from_db(db_path: Path, tables: Sequence[str], flight_id: str) -> List[Dict[str, Any]]:
    if not db_path.exists():
        return []

    with connect(db_path) as conn:
        for table in tables:
            try:
                rows = conn.execute(
                    f"""
                    SELECT flight_id, timestamp, lat, lon, alt, heading, gspeed, vspeed, track, squawk, callsign, source
                    FROM {table}
                    WHERE flight_id = ?
                    ORDER BY timestamp ASC
                    """,
                    (flight_id,),
                ).fetchall()
            except sqlite3.Error:
                rows = []

            if rows:
                return [dict(row) for row in rows]
    return []


def load_track_points(flight_id: str) -> List[Dict[str, Any]]:
    # Try the training dataset first (where feedback saves tracks)
    track_points = _query_track_from_db(TRAINING_DB, ["anomalous_tracks", "flight_tracks"], flight_id)
    if track_points:
        return track_points

    # Fallback to research DB (live anomalies/normal tracks)
    track_points = _query_track_from_db(RESEARCH_DB, ["anomalies_tracks", "normal_tracks"], flight_id)
    if track_points:
        return track_points

    # Fallback to last.db (historic merged data)
    track_points = _query_track_from_db(LAST_DB, ["anomalous_tracks", "flight_tracks"], flight_id)
    return track_points


def _load_anomaly_report_from(db_path: Path, flight_id: str) -> Optional[Dict[str, Any]]:
    if not db_path.exists():
        return None

    with connect(db_path) as conn:
        try:
            row = conn.execute(
                """
                SELECT flight_id, timestamp, is_anomaly, severity_cnn, severity_dense, full_report
                FROM anomaly_reports
                WHERE flight_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (flight_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            return None

        if not row:
            return None

        raw_report = row["full_report"]
        parsed_report: Optional[Dict[str, Any]] = None
        if isinstance(raw_report, (str, bytes)):
            try:
                parsed_report = json.loads(raw_report)
            except Exception:
                parsed_report = None
        elif isinstance(raw_report, dict):
            parsed_report = raw_report

        return {
            "flight_id": row["flight_id"],
            "timestamp": row["timestamp"],
            "is_anomaly": bool(row["is_anomaly"]),
            "severity_cnn": row["severity_cnn"],
            "severity_dense": row["severity_dense"],
            "full_report": parsed_report,
            "full_report_raw": raw_report,
        }


def load_anomaly_report(flight_id: str, feedback_report_json: Optional[str] = None) -> Optional[Dict[str, Any]]:
    # 1. Prefer report stored in feedback DB (snapshot at time of feedback)
    if feedback_report_json:
        try:
            parsed = json.loads(feedback_report_json)
            # Ensure it has necessary fields or wrap it
            if isinstance(parsed, dict):
                # If stored as full report, it might lack 'is_anomaly' top level keys if those were outside json
                # But usually we store the full json blob.
                # Let's reconstruct a "db row" like structure if possible, or just return it as full_report
                # The pipeline report structure usually has 'summary', 'layer_1_rules', etc.
                summary = parsed.get("summary", {})
                return {
                    "flight_id": flight_id,
                    "timestamp": 0, # Unknown if not in JSON
                    "is_anomaly": summary.get("is_anomaly"),
                    "severity_cnn": parsed.get("severity_cnn"),
                    "severity_dense": parsed.get("severity_dense"),
                    "full_report": parsed,
                    "full_report_raw": feedback_report_json,
                }
        except Exception:
            pass

    # 2. Prefer research DB; fallback to live anomalies DB
    report = _load_anomaly_report_from(RESEARCH_DB, flight_id)
    if report:
        return report
    return _load_anomaly_report_from(LIVE_ANOMALIES_DB, flight_id)


def _safe_join(values: Iterable[Any]) -> str:
    return ", ".join(str(v) for v in values if v is not None and str(v) != "")


def flatten_rules(full_report: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Extract rule-related columns plus a per-rule row list.
    Returns (aggregate_columns, rule_rows).
    """
    if not full_report:
        return {}, []

    rules_layer = full_report.get("layer_1_rules") or {}
    rule_report = rules_layer.get("report") or {}
    matched_rules: List[Dict[str, Any]] = rule_report.get("matched_rules") or []
    evaluations: List[Dict[str, Any]] = rule_report.get("evaluations") or []

    aggregate = {
        "rules_status": rules_layer.get("status"),
        "rules_triggers": _safe_join(rules_layer.get("triggers") or []),
        "matched_rule_ids": _safe_join([r.get("id") for r in matched_rules]),
        "matched_rule_names": _safe_join([r.get("name") for r in matched_rules]),
        "matched_rule_categories": _safe_join([r.get("category") for r in matched_rules]),
    }

    rule_rows: List[Dict[str, Any]] = []
    for r in evaluations:
        rule_rows.append(
            {
                "rule_id": r.get("id"),
                "rule_name": r.get("name"),
                "category": r.get("category"),
                "severity": r.get("severity"),
                "matched": 1 if r.get("matched") else 0,
                "summary": _coerce_text(r.get("summary")),
                "details": _coerce_text(r.get("details")),
            }
        )

    return aggregate, rule_rows


def _coerce_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _serialize_full_report(report: Optional[Dict[str, Any]]) -> Optional[str]:
    if not report:
        return None

    content = report.get("full_report")
    if content is None:
        content = report.get("full_report_raw")

    if content is None:
        return None

    if isinstance(content, bytes):
        return content.decode("utf-8", errors="ignore")
    if isinstance(content, str):
        return content

    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def insert_report(
    conn: sqlite3.Connection,
    feedback: FeedbackRow,
    report: Optional[Dict[str, Any]],
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Insert into anomaly_reports and return (report_id, aggregated_rule_columns).
    """
    full_report = report.get("full_report") if report else None
    flat_rules, rule_rows = flatten_rules(full_report)
    summary = (full_report or {}).get("summary") or {}

    conn.execute(
        """
        INSERT INTO anomaly_reports (
            flight_id, feedback_id, feedback_timestamp, user_label,
            comments, model_version, rule_id, other_details,
            anomaly_timestamp, pipeline_is_anomaly,
            severity_cnn, severity_dense, confidence_score,
            summary_triggers, rules_status, rules_triggers,
            matched_rule_ids, matched_rule_names, matched_rule_categories,
            full_report_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            feedback.flight_id,
            feedback.id,
            feedback.timestamp,
            feedback.user_label,
            feedback.comments,
            feedback.model_version,
            feedback.rule_id,
            feedback.other_details,
            report["timestamp"] if report else None,
            summary.get("is_anomaly"),
            (report or {}).get("severity_cnn"),
            (report or {}).get("severity_dense"),
            summary.get("confidence_score"),
            _safe_join(summary.get("triggers") or []),
            flat_rules.get("rules_status"),
            flat_rules.get("rules_triggers"),
            flat_rules.get("matched_rule_ids"),
            flat_rules.get("matched_rule_names"),
            flat_rules.get("matched_rule_categories"),
            _serialize_full_report(report),
        ),
    )

    report_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    return report_id, rule_rows


def insert_rule_rows(conn: sqlite3.Connection, report_id: int, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO rule_matches (
            report_id, rule_id, rule_name, category, severity, matched, summary, details
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                report_id,
                r.get("rule_id"),
                r.get("rule_name"),
                r.get("category"),
                r.get("severity"),
                r.get("matched"),
                r.get("summary"),
                r.get("details"),
            )
            for r in rows
        ],
    )


def insert_tracks(conn: sqlite3.Connection, feedback: FeedbackRow, points: List[Dict[str, Any]]) -> None:
    if not points:
        return
    conn.executemany(
        """
        INSERT OR REPLACE INTO flight_tracks (
            flight_id, timestamp, lat, lon, alt, heading, gspeed, vspeed, track, squawk, callsign, source, user_label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                p.get("flight_id"),
                p.get("timestamp"),
                p.get("lat"),
                p.get("lon"),
                p.get("alt"),
                p.get("heading"),
                p.get("gspeed"),
                p.get("vspeed"),
                p.get("track"),
                p.get("squawk"),
                p.get("callsign"),
                p.get("source"),
                feedback.user_label,
            )
            for p in points
        ],
    )


def migrate() -> None:
    feedback_rows = load_feedback_rows()
    ensure_present_schema()

    with connect(PRESENT_DB) as dest_conn:
        for fb in feedback_rows:
            track_points = load_track_points(fb.flight_id)
            report = load_anomaly_report(fb.flight_id, fb.full_report_json)

            report_id, rule_rows = insert_report(dest_conn, fb, report)
            insert_rule_rows(dest_conn, report_id, rule_rows)
            insert_tracks(dest_conn, fb, track_points)

        dest_conn.commit()


if __name__ == "__main__":
    migrate()
    print(f"Done. Exported feedback flights to {PRESENT_DB}")

