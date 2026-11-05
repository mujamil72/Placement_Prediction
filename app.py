from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="Placement Prediction API")

artifact = joblib.load("model.pkl")
model = artifact["model"]
FEATURES = artifact["features"]


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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: StudentInput):
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
    return {"prediction": "Placed" if pred == 1 else "NotPlaced",
            "confidence": round(prob, 4)}
