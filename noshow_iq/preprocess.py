import pandas as pd


def load_and_clean(path):
    df = pd.read_csv(path)

    df.columns = [c.strip().lower().replace('-', '_') for c in df.columns]

    if 'no-show' in df.columns:
        df.rename(columns={'no-show': 'no_show'}, inplace=True)

    df['no_show'] = df['no_show'].map({'Yes': 1, 'No': 0})

    df = df[df['age'] >= 0]
    df = df[df['age'] <= 115]

    df['scheduledday'] = pd.to_datetime(df['scheduledday'])
    df['appointmentday'] = pd.to_datetime(df['appointmentday'])

    df['days_in_advance'] = (
        df['appointmentday'] - df['scheduledday']
    ).dt.days
    df['days_in_advance'] = df['days_in_advance'].clip(lower=0)

    df['appointment_hour'] = df['scheduledday'].dt.hour

    df.drop(
        columns=['patientid', 'appointmentid',
                 'scheduledday', 'appointmentday',
                 'neighbourhood'],
        inplace=True
    )

    df['gender'] = df['gender'].map({'F': 0, 'M': 1})

    df = df.dropna()

    return df


def get_features_target(df):
    X = df.drop(columns=['no_show'])
    y = df['no_show']
    return X, y
