import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/placedata.csv")

df["ExtracurricularActivities"] = df["ExtracurricularActivities"].map({"Yes": 1, "No": 0})
df["PlacementTraining"] = df["PlacementTraining"].map({"Yes": 1, "No": 0})
y = df["PlacementStatus"].map({"Placed": 1, "NotPlaced": 0})
X = df.drop(columns=["StudentID", "PlacementStatus"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred, target_names=["NotPlaced", "Placed"]))

joblib.dump({"model": model, "features": list(X.columns)}, "model.pkl")
print("Saved model.pkl with features:", list(X.columns))
