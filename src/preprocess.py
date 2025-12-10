import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def clean_url(url):
    url = url.lower().strip()

    # Remove fragment
    url = re.sub(r'#.*$', '', url)

    # Remove default ports
    url = re.sub(r':80/', '/', url)
    url = re.sub(r':443/', '/', url)

    # Remove multiple slashes
    url = re.sub(r'//+', '/', url)

    return url

def load_and_clean(path):
    df = pd.read_csv(path)

    df = df.drop_duplicates()
    df = df.dropna()

    df['clean_url'] = df['url'].apply(clean_url)
    return df

def shannon_entropy(s):
    probabilities = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * np.log2(p) for p in probabilities)

def extract_features(df):
    hosts = df['clean_url'].apply(lambda x: urlparse(x).netloc)
    paths = df['clean_url'].apply(lambda x: urlparse(x).path)

    df['url_length'] = df['clean_url'].str.len()
    df['host_length'] = hosts.str.len()
    df['path_length'] = paths.str.len()

    df['digit_count'] = df['clean_url'].str.count(r'\d')
    df['special_count'] = df['clean_url'].str.count(r'[!@#$%^&*()\-_=+{};:,<.>?/]')
    df['subdomain_count'] = hosts.str.count(r'\.')

    df['digit_ratio'] = df['digit_count'] / df['url_length']
    df['special_ratio'] = df['special_count'] / df['url_length']

    df['url_entropy'] = df['clean_url'].apply(shannon_entropy)

    df['has_ip'] = df['clean_url'].str.contains(r'\d+\.\d+\.\d+\.\d+').astype(int)
    df['has_at'] = df['clean_url'].str.contains('@').astype(int)
    df['has_https'] = df['clean_url'].str.startswith("https").astype(int)

    feature_cols = [
        'url_length','host_length','path_length',
        'digit_count','special_count','subdomain_count',
        'digit_ratio','special_ratio','url_entropy',
        'has_ip','has_at','has_https'
    ]

    return df, feature_cols

def scale_and_split(df, feature_cols):
    X = df[feature_cols]
    y = df['type']  # (benign, phishing, malware, defacement)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sm = SMOTE(random_state=42)
    X_bal, y_bal = sm.fit_resample(X_scaled, y)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_bal, y_bal, test_size=0.30, random_state=42, stratify=y_bal
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
