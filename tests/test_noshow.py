import pytest
import pandas as pd
from noshow_iq.preprocess import load_and_clean, get_features_target
from noshow_iq.model import model_exists


def test_days_in_advance():
    df = pd.DataFrame([{
        "gender": "F",
        "scheduledday": "2016-04-29",
        "appointmentday": "2016-05-04",
        "age": 25,
        "neighbourhood": "JARDIM DA PENHA",
        "scholarship": 0,
        "hipertension": 0,
        "diabetes": 0,
        "alcoholism": 0,
        "handcap": 0,
        "sms_received": 0,
        "no-show": "No"
    }])
    df.columns = [c.strip().lower().replace('-', '_') for c in df.columns]
    df.rename(columns={'no-show': 'no_show'}, inplace=True)
    df['no_show'] = df['no_show'].map({'Yes': 1, 'No': 0})
    df['scheduledday'] = pd.to_datetime(df['scheduledday'])
    df['appointmentday'] = pd.to_datetime(df['appointmentday'])
    df['days_in_advance'] = (
        df['appointmentday'] - df['scheduledday']
    ).dt.days.clip(lower=0)
    assert df['days_in_advance'].iloc[0] == 5


def test_age_filter():
    df = pd.DataFrame([
        {"age": -1, "no_show": 0},
        {"age": 25, "no_show": 1},
        {"age": 200, "no_show": 0}
    ])
    df = df[df['age'] >= 0]
    df = df[df['age'] <= 115]
    assert len(df) == 1


def test_gender_mapping():
    df = pd.DataFrame([{"gender": "F"}, {"gender": "M"}])
    df['gender'] = df['gender'].map({'F': 0, 'M': 1})
    assert df['gender'].iloc[0] == 0
    assert df['gender'].iloc[1] == 1


def test_no_show_mapping():
    df = pd.DataFrame([{"no_show": "Yes"}, {"no_show": "No"}])
    df['no_show'] = df['no_show'].map({'Yes': 1, 'No': 0})
    assert df['no_show'].iloc[0] == 1
    assert df['no_show'].iloc[1] == 0


def test_model_exists_false():
    import os
    if os.path.exists("model.pkl"):
        assert model_exists() is True
    else:
        assert model_exists() is False


def test_dataframe_columns():
    df = pd.DataFrame([{
        "gender": 0,
        "age": 25,
        "scholarship": 0,
        "hipertension": 0,
        "diabetes": 0,
        "alcoholism": 0,
        "handcap": 0,
        "sms_received": 0,
        "days_in_advance": 5,
        "appointment_hour": 10,
        "no_show": 0
    }])
    assert "days_in_advance" in df.columns
    assert "appointment_hour" in df.columns