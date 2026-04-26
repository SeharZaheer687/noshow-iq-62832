import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE


MODEL_PATH = "model.pkl"


def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Handle class imbalance with SMOTE
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    report = evaluate(model, X_test, y_test)
    return model, report


def predict(X):
    model = joblib.load(MODEL_PATH)
    prob = model.predict_proba(X)[0][1]
    risk = "high" if prob >= 0.5 else "low"
    return risk, float(prob)


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred, output_dict=True
    )
    return report


def model_exists():
    return os.path.exists(MODEL_PATH)
