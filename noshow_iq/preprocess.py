import pandas as pd
from datetime import datetime


def load_and_clean(path):
    df = pd.read_csv(path)

    # Fix column names
    df.columns = [c.strip().lower().replace('-', '_') for c in df.columns]

    # Rename target column
    if 'no-show' in df.columns:
        df.rename(columns={'no-show': 'no_show'}, inplace=True)

    # Fix target values
    df['no_show'] = df['no_show'].map({'Yes': 1, 'No': 0})

    # Remove bad ages
    df = df[df['age'] >= 0]
    df = df[df['age'] <= 115]

    # Convert dates
    df['scheduledday'] = pd.to_datetime(df['scheduledday'])
    df['appointmentday'] = pd.to_datetime(df['appointmentday'])

    # New feature 1: days in advance
    df['days_in_advance'] = (
        df['appointmentday'] - df['scheduledday']
    ).dt.days
    df['days_in_advance'] = df['days_in_advance'].clip(lower=0)

    # New feature 2: appointment hour
    df['appointment_hour'] = df['scheduledday'].dt.hour

    # Drop useless columns
    df.drop(
        columns=['patientid', 'appointmentid',
                 'scheduledday', 'appointmentday'],
        inplace=True
    )

    # Fix gender
    df['gender'] = df['gender'].map({'F': 0, 'M': 1})

    return df


def get_features_target(df):
    X = df.drop(columns=['no_show'])
    y = df['no_show']
    return X, y