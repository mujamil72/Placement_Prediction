import logging
import time
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# ---------- Logging (Practical-11) ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("placement_api")

# ---------- Prometheus metrics (Practical-12) ----------
HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests received by the API",
    ["method", "path", "status"],
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total prediction attempts",
    ["result"],
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the model is loaded",
)

app = FastAPI(title="Placement Prediction API")

# ---------- Model loading ----------
try:
    artifact = joblib.load("model.pkl")
    model = artifact["model"]
    FEATURES = artifact["features"]
    MODEL_LOADED.set(1)
    logger.info("event=model_loaded model_file=model.pkl")
except Exception:
    MODEL_LOADED.set(0)
    logger.exception("event=model_load_failed model_file=model.pkl")
    raise


class StudentInput(BaseModel):
    CGPA: float
    Internships: int
    Projects: int
    Workshops_Certifications: int = Field(alias="Workshops/Certifications")
    AptitudeTestScore: int
    SoftSkillsRating: float
    ExtracurricularActivities: str  # "Yes" or "No"
    PlacementTraining: str          # "Yes" or "No"
    SSC_Marks: int
    HSC_Marks: int

    model_config = {"populate_by_name": True}


# ---------- Middleware: logging + metrics ----------
@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    path = request.url.path
    method = request.method
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        duration = time.perf_counter() - start_time
        if path != "/metrics":
            HTTP_REQUESTS.labels(method=method, path=path, status="500").inc()
            HTTP_LATENCY.labels(method=method, path=path).observe(duration)
        logger.exception(
            "event=request_failed request_id=%s method=%s path=%s",
            request_id, method, path,
        )
        raise

    duration = time.perf_counter() - start_time

    # Do not count Prometheus' own scrape requests in user-traffic metrics
    if path != "/metrics":
        HTTP_REQUESTS.labels(method=method, path=path, status=str(status)).inc()
        HTTP_LATENCY.labels(method=method, path=path).observe(duration)

    level = logging.WARNING if status >= 400 else logging.INFO
    logger.log(
        level,
        "event=request_completed request_id=%s method=%s path=%s status=%s duration=%.4f",
        request_id, method, path, status, duration,
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"message": "Placement Prediction API is running", "version": "1.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(data: StudentInput):
    try:
        row = {
            "CGPA": data.CGPA,
            "Internships": data.Internships,
            "Projects": data.Projects,
            "Workshops/Certifications": data.Workshops_Certifications,
            "AptitudeTestScore": data.AptitudeTestScore,
            "SoftSkillsRating": data.SoftSkillsRating,
            "ExtracurricularActivities": 1 if data.ExtracurricularActivities.lower() == "yes" else 0,
            "PlacementTraining": 1 if data.PlacementTraining.lower() == "yes" else 0,
            "SSC_Marks": data.SSC_Marks,
            "HSC_Marks": data.HSC_Marks,
        }
        df = pd.DataFrame([row])[FEATURES]
        pred = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[0][pred])

        PREDICTIONS.labels(result="success").inc()
        logger.info("event=prediction_succeeded")

        return {
            "prediction": "Placed" if pred == 1 else "NotPlaced",
            "confidence": round(prob, 4),
        }
    except Exception:
        PREDICTIONS.labels(result="error").inc()
        logger.exception("event=prediction_failed")
        raise