import os
import json
import logging
import io
import csv
from typing import Dict, Any, List
from datetime import datetime

# Try to import openai, but handle if missing
try:
    import openai
except ImportError:
    openai = None

logger = logging.getLogger(__name__)

def analyze_anomaly_with_llm(
    anomaly_report: Dict[str, Any],
    flight_data_summary: Dict[str, Any],
    glitch_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Uses an LLM (simulated or real) to analyze the anomaly.
    """

    api_key = os.getenv("OPENAI_API_KEY", "")

    # Generate CSV string from full trajectory
    full_trajectory = flight_data_summary.get("full_trajectory", [])
    csv_buffer = io.StringIO()
    if full_trajectory:
        fieldnames = ["ts", "lat", "lon", "alt", "spd", "hdg", "vspd"]
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        for p in full_trajectory:
            # Ensure keys match fieldnames, fill missing with empty string
            row = {k: p.get(k, "") for k in fieldnames}
            writer.writerow(row)

    csv_data = csv_buffer.getvalue()

    # Construct the prompt content
    # We remove full_trajectory from json dump to save tokens/clutter, since we provide CSV
    summary_for_json = flight_data_summary.copy()
    if "full_trajectory" in summary_for_json:
        del summary_for_json["full_trajectory"]

    prompt_data = {
        "anomaly_report": anomaly_report,
        "glitch_analysis": glitch_analysis,
        "flight_context": summary_for_json
    }

    system_prompt = """
    You are an expert aviation safety analyst. You are reviewing an anomaly detected by an automated system.
    Your goal is to determine if this is a REAL operational anomaly (e.g., evasive maneuver, emergency descent, holding pattern) 
    or a FALSE ALARM caused by data noise/glitches (e.g., tracking errors, frozen positions, spikes).
    rethink the points that are reported anomaly and look at it from a human vision to see if its an anomaly or the flight is looking ok overall. 
    
    holding in the air or landing in a non-regular or unexpected manner is classified as an anomaly.
    
    You have:
    1. The automated anomaly report.
    2. A glitch analysis report (looking for data quality issues).
    3. Flight data context.
    
    Output a JSON object with the following fields:
    - "explanation": A short, human-readable explanation of what likely happened.
    - "logical_judgment": "real anomaly" or "normal but noisy".
    - "logical_anomaly_score": A score from 0 to 100 reflecting confidence it is a REAL anomaly.
    - "is_anomaly": boolean (true if real anomaly).
    - "danger_level": "low", "medium", "high".
    - "reasoning": Detailed technical reasoning.
    - "should_escalate": boolean.
    
    You should think like a real human and look and the data from a logical perspective.

    Focus on the physics.  90 degree turns in 1 second are glitches. 
    """

    user_message = f"""Analyze this flight event.

METADATA (JSON):
{json.dumps(prompt_data, default=str)}

"""

    # If we have an API key and library, call GPT-4o (closest to '5' available)
    if api_key and openai:
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # Fallthrough to mock
            pass

    # --- IMPROVED MOCK LOGIC (when no API key) ---

    glitch_score = glitch_analysis.get("glitch_score", 0)
    summary = anomaly_report.get("summary", {})
    triggers = summary.get("triggers", [])

    # 1. High Glitch Score -> "Normal but noisy"
    if glitch_score > 0.25:  # Lowered threshold slightly
        return {
            "explanation": f"High likelihood of data artifacts (Score: {glitch_score:.2f}). Speed spikes/gaps detected.",
            "logical_judgment": "normal but noisy",
            "logical_anomaly_score": max(0, 10 - int(glitch_score*10)),
            "is_anomaly": False,
            "danger_level": "low",
            "reasoning": f"Glitch analyzer flagged significant data quality issues. Physics violations observed.",
            "should_escalate": False
        }

    # 2. Specific Rule Triggers -> Tailored Explanation
    if "Rules" in triggers:
        rules_report = anomaly_report.get("layer_1_rules", {}).get("report", {})
        matched_rules = rules_report.get("matched_rules", [])
        rule_names = [r.get("name", "Unknown Rule") for r in matched_rules]
        rule_str = ", ".join(rule_names)

        return {
            "explanation": f"Flight violated safety rules: {rule_str}.",
            "logical_judgment": "real anomaly",
            "logical_anomaly_score": 95,
            "is_anomaly": True,
            "danger_level": "high",
            "reasoning": f"Hard coded safety rules were triggered. This indicates a clear deviation from standard procedure.",
            "should_escalate": True
        }

    # 3. ML Model Triggers -> Tailored Explanation
    model_triggers = [t for t in triggers if t != "Rules"]
    if model_triggers:
        models_str = ", ".join(model_triggers)
        return {
            "explanation": f"Abnormal pattern detected by {models_str} models.",
            "logical_judgment": "real anomaly",
            "logical_anomaly_score": 75,
            "is_anomaly": True,
            "danger_level": "medium",
            "reasoning": f"Multiple ML layers ({models_str}) agreed on the anomaly. Data quality is clean (Glitch Score: {glitch_score:.2f}).",
            "should_escalate": True
        }

    # 4. Fallback for low confidence or empty triggers
    return {
        "explanation": "Slight deviation detected, but likely within normal operational bounds.",
        "logical_judgment": "normal but noisy",
        "logical_anomaly_score": 35,
        "is_anomaly": False,
        "danger_level": "low",
        "reasoning": "Anomaly signal was weak and data quality is clean. No specific safety rules were violated.",
        "should_escalate": False
    }
