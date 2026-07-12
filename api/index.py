import os
import sys
import json
import base64
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
FALLBACK_MODELS = list(dict.fromkeys([DEFAULT_MODEL, "gemini-flash-latest", "gemini-flash-lite-latest"]))

RECONSTRUCT_SCHEMA = {
    "type": "object",
    "properties": {
        "columns": {"type": "array", "items": {"type": "string"}},
        "column_types": {
            "type": "array",
            "items": {"type": "string", "enum": ["numeric", "categorical"]},
        },
        "rows": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"}},
        },
    },
    "required": ["columns", "column_types", "rows"],
}


def _detect_mime(audio_bytes: bytes) -> str:
    if audio_bytes[:4] == b"RIFF":
        return "audio/wav"
    if audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        return "audio/mpeg"
    if audio_bytes[:4] == b"OggS":
        return "audio/ogg"
    if audio_bytes[4:8] == b"ftyp":
        return "audio/mp4"
    if audio_bytes[:4] == b"\x1a\x45\xdf\xa3":
        return "audio/webm"
    if audio_bytes[:4] == b"fLaC":
        return "audio/flac"
    return "audio/wav"


PROMPT = """Listen to this audio carefully - it is someone reading aloud a small
structured dataset (a table), possibly in Korean, row by row. Each row repeats
the SAME small set of attribute names (e.g. "height", "weight") followed by a
value for that row.

Reconstruct the exact table as JSON:
- "columns": the list of DISTINCT attribute names that repeat across every
  row, in the order first spoken. This is almost always a SMALL number (often
  just 1-3) - do NOT create a new column for a value, a filler word, or a
  one-off phrase. A column name should be something you hear repeated with a
  new value on every single row, not a word that appears once.
- "column_types": whether each column is "numeric" or "categorical", in the
  same order as columns.
- "rows": the list of rows, each row being a list of values in the same
  column order (one entry per column, nothing extra). Convert any numbers
  spoken as words into plain numerals. The number of rows should match how
  many times the full set of attributes was repeated in the audio.

Be precise - do not guess or invent values not present in the audio, and do
not let stray/filler words become extra columns or extra row entries."""


def _reconstruct_table(audio_bytes: bytes) -> dict:
    mime_type = _detect_mime(audio_bytes)
    last_error = None
    for model_name in FALLBACK_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    PROMPT,
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RECONSTRUCT_SCHEMA,
                    temperature=0,
                ),
            )
            table = json.loads(response.text)
            print(f"[q6-debug] model={model_name} reconstructed_table={json.dumps(table)}", file=sys.stderr, flush=True)
            return table
        except Exception as e:
            last_error = e
            if "not found" in str(e).lower() or "unsupported" in str(e).lower():
                continue
            raise
    raise RuntimeError(f"All models failed: {last_error}")


def _build_dataframe(table: dict):
    columns = table["columns"]
    types_ = table["column_types"]
    rows = table["rows"]

    df = pd.DataFrame(rows, columns=columns)

    numeric_cols = []
    categorical_cols = []
    for col, col_type in zip(columns, types_):
        if col_type == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return df, numeric_cols, categorical_cols


def _stats_for_audio(audio_bytes: bytes) -> dict:
    table = _reconstruct_table(audio_bytes)
    df, numeric_cols, categorical_cols = _build_dataframe(table)

    def _mode_of(series):
        m = series.mode()
        if len(m) == 0:
            return None
        val = m.iloc[0]
        return val.item() if hasattr(val, "item") else val

    mean = {c: float(df[c].mean()) for c in numeric_cols}
    std = {c: float(df[c].std()) for c in numeric_cols}
    variance = {c: float(df[c].var()) for c in numeric_cols}
    min_ = {c: float(df[c].min()) for c in numeric_cols}
    max_ = {c: float(df[c].max()) for c in numeric_cols}
    median = {c: float(df[c].median()) for c in numeric_cols}
    mode = {c: _mode_of(df[c]) for c in df.columns}
    range_ = {c: float(df[c].max() - df[c].min()) for c in numeric_cols}
    allowed_values = {c: sorted(df[c].dropna().unique().tolist()) for c in categorical_cols}
    value_range = {c: [float(df[c].min()), float(df[c].max())] for c in numeric_cols}

    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr()
        correlation = corr_df.values.tolist()
    else:
        correlation = []

    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "mean": mean,
        "std": std,
        "variance": variance,
        "min": min_,
        "max": max_,
        "median": median,
        "mode": mode,
        "range": range_,
        "allowed_values": allowed_values,
        "value_range": value_range,
        "correlation": correlation,
    }


@app.get("/")
def root():
    return {"status": "ok"}


async def _handle(request: Request):
    payload = await request.json()
    audio_base64 = payload.get("audio_base64", "")
    audio_bytes = base64.b64decode(audio_base64)
    return _stats_for_audio(audio_bytes)


# The spec doesn't name a fixed sub-path - it says the grader POSTs to
# whatever base URL is submitted - so accept the request on both "/" and
# a couple of likely alias paths.
@app.post("/")
async def answer_root(request: Request):
    return await _handle(request)


@app.post("/answer")
async def answer_path(request: Request):
    return await _handle(request)


@app.post("/api")
async def answer_api(request: Request):
    return await _handle(request)
