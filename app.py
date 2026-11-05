import logging
import sys
import time
import uuid

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from pydantic import BaseModel, Field
import joblib
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger("placement-api")

app = FastAPI(title="Placement Prediction API")

MODEL_FILE = "model.pkl"

try:
    artifact = joblib.load(MODEL_FILE)
    model = artifact["model"]
    FEATURES = artifact["features"]
    logger.info("event=model_loaded model_file=%s", MODEL_FILE)
except Exception:
    logger.exception("event=model_load_failed model_file=%s", MODEL_FILE)
    raise


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(
            level,
            "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        response.headers["X-Request-ID"] = request_id
        return response

    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "event=http_request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    fields = [
        ".".join(str(part) for part in error.get("loc", []))
        for error in exc.errors()
    ]

    logger.warning(
        "event=validation_error path=%s field_locations=%s error_count=%s",
        request.url.path,
        fields,
        len(exc.errors()),
    )

    return await request_validation_exception_handler(request, exc)


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


@app.get("/")
def root():
    return {"message": "Placement Prediction API is running", "version": "1.1"}


@app.get("/health")
def health():
    return {"status": "ok"}


# Add only non-sensitive features here if input-value logging is required.
SAFE_INPUT_FIELDS = set()


@app.post("/predict")
def predict(data: StudentInput):
    try:
        payload = data.model_dump()

        safe_input = {
            key: value
            for key, value in payload.items()
            if key in SAFE_INPUT_FIELDS
        }

        logger.info(
            "event=prediction_request feature_names=%s safe_input=%s",
            sorted(payload.keys()),
            safe_input,
        )

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

        result = "Placed" if pred == 1 else "NotPlaced"
        logger.info("event=prediction_success prediction=%s", result)

        return {"prediction": result, "confidence": round(prob, 4)}

    except Exception:
        logger.exception("event=prediction_failed")
        raise HTTPException(status_code=500, detail="Prediction failed")