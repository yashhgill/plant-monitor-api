from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import deque
from datetime import datetime
import httpx
import os
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory store ───────────────────────────
history = deque(maxlen=100)
latest  = {"temperature": 0, "humidity": 0, "fan": False, "pump": False, "auto": True, "ts": ""}
pending_command = {"fan": None, "pump": None, "auto": None}

# ── Active thresholds ─────────────────────────
active_thresholds = {
    "temp_high":  25.0,
    "temp_low":   23.0,
    "humid_low":  40.0,
    "humid_high": 55.0,
    "plant":      "",
    "advice":     "",
}

GROQ_MODEL = "openai/gpt-oss-20b"

# ── Models ────────────────────────────────────
class SensorData(BaseModel):
    temperature: float
    humidity:    float
    fan:         bool
    pump:        bool
    auto:        bool

class Command(BaseModel):
    fan:  bool | None = None
    pump: bool | None = None
    auto: bool | None = None

class ThresholdOverride(BaseModel):
    temp_high:  float
    temp_low:   float
    humid_low:  float
    humid_high: float

# ── Sensor data ───────────────────────────────
@app.post("/data")
async def receive_data(data: SensorData):
    global latest
    ts = datetime.now().strftime("%H:%M:%S")
    entry = {**data.dict(), "ts": ts}
    latest = entry
    history.append(entry)
    return {"ok": True}

@app.get("/data")
async def get_data():
    return {
        "latest":     latest,
        "history":    list(history),
        "thresholds": active_thresholds,
    }

# ── Commands ──────────────────────────────────
@app.post("/control")
async def set_control(cmd: Command):
    global pending_command
    if cmd.fan  is not None: pending_command["fan"]  = cmd.fan
    if cmd.pump is not None: pending_command["pump"] = cmd.pump
    if cmd.auto is not None: pending_command["auto"] = cmd.auto
    return {"ok": True}

@app.get("/control")
async def get_control():
    global pending_command
    cmd = dict(pending_command)
    pending_command = {"fan": None, "pump": None, "auto": None}
    return cmd

# ── Thresholds ────────────────────────────────
@app.get("/thresholds")
async def get_thresholds():
    return active_thresholds

@app.post("/thresholds")
async def set_thresholds(t: ThresholdOverride):
    global active_thresholds
    active_thresholds.update(t.dict())
    return {"ok": True, "thresholds": active_thresholds}

# ── Groq AI advice ────────────────────────────
@app.get("/ai-advice")
async def ai_advice(plant: str = "tomato"):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "") or "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"

    temp  = latest.get("temperature", 0)
    humid = latest.get("humidity",    0)

    prompt = f"""You are a plant care expert AI controlling an automated greenhouse system.

Current environment: Temperature {temp}°C, Humidity {humid}%.
Plant: {plant}

Respond ONLY with a valid JSON object. No markdown, no backticks, no explanation.
Use exactly these keys:
{{
  "temp_high": <max comfortable °C as number>,
  "temp_low":  <min comfortable °C as number>,
  "humid_low": <minimum humidity % as number>,
  "humid_high":<maximum humidity % as number>,
  "advice":    "<2 sentence care tip for this plant and why these thresholds>"
}}"""

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       GROQ_MODEL,
                    "temperature": 0.3,
                    "messages": [
                        {
                            "role":    "system",
                            "content": "You are a plant care expert. Always respond with valid JSON only. No markdown, no extra text."
                        },
                        {
                            "role":    "user",
                            "content": prompt
                        }
                    ]
                }
            )
            r.raise_for_status()
            data    = r.json()
            raw     = data["choices"][0]["message"]["content"].strip()

            # Strip any accidental markdown fences
            raw = raw.strip("```json").strip("```").strip()
            parsed  = json.loads(raw)

            # Validate required keys
            for k in ["temp_high", "temp_low", "humid_low", "humid_high", "advice"]:
                if k not in parsed:
                    raise ValueError(f"Missing key: {k}")

            # Save as active thresholds
            global active_thresholds
            active_thresholds.update({
                "temp_high":  float(parsed["temp_high"]),
                "temp_low":   float(parsed["temp_low"]),
                "humid_low":  float(parsed["humid_low"]),
                "humid_high": float(parsed["humid_high"]),
                "plant":      plant,
                "advice":     parsed["advice"],
            })

            return {
                "thresholds": active_thresholds,
                "advice":     parsed["advice"],
            }

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"AI returned invalid JSON: {e}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Groq API error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
