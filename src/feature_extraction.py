# feature_extraction.py
"""
Feature extraction utilities for lexical malicious URL detection.

Typical usage in preprocess.py:

    from feature_extraction import add_lexical_features

    df = load_raw_dataset(...)
    df, feature_cols = add_lexical_features(df, url_col="url")

"""

import re
import numpy as np
import pandas as pd
from urllib.parse import urlparse


def clean_url(url: str) -> str:
    """Canonicalize a URL string: lowercase, drop fragments, strip default ports, collapse slashes."""
    if not isinstance(url, str):
        return ""

    url = url.strip().lower()

    # Remove fragment (#...)
    url = re.sub(r"#.*$", "", url)

    # Strip common default ports
    url = re.sub(r":80(/|$)", r"/", url)
    url = re.sub(r":443(/|$)", r"/", url)

    # Collapse multiple slashes (but keep protocol '://')
    if "://" in url:
        scheme, rest = url.split("://", 1)
        rest = re.sub(r"/{2,}", "/", rest)
        url = scheme + "://" + rest
    else:
        url = re.sub(r"/{2,}", "/", url)

    return url


def shannon_entropy(s: str) -> float:
    """Compute Shannon entropy for a string."""
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return float(-sum(p * np.log2(p) for p in probs))


def _safe_ratio(numer, denom) -> float:
    if denom == 0:
        return 0.0
    return float(numer) / float(denom)


def add_lexical_features(df: pd.DataFrame, url_col: str = "url"):
    """
    Add lexical / structural URL features to a DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must contain a column with raw URLs.
    url_col : str
        Column name that stores the URLs.

    Returns
    -------
    df : DataFrame
        Same DF with new feature columns.
    feature_cols : list[str]
        Names of the numeric feature columns to feed into models.
    """
    if url_col not in df.columns:
        raise ValueError(f"Expected column '{url_col}' not found in DataFrame")

    # Canonical URL
    df["clean_url"] = df[url_col].astype(str).apply(clean_url)

    # Parsed components
    hosts = df["clean_url"].apply(lambda x: urlparse(x).netloc)
    paths = df["clean_url"].apply(lambda x: urlparse(x).path)
    queries = df["clean_url"].apply(lambda x: urlparse(x).query)

    # Length-based features
    df["url_length"] = df["clean_url"].str.len()
    df["host_length"] = hosts.str.len()
    df["path_length"] = paths.str.len()
    df["query_length"] = queries.str.len()

    # Count-based features
    df["digit_count"] = df["clean_url"].str.count(r"\d")
    df["special_count"] = df["clean_url"].str.count(r"[!@#$%^&*()\-_=+{};:,<.>?/]")
    df["dot_count"] = df["clean_url"].str.count(r"\.")
    df["subdomain_count"] = hosts.str.count(r"\.")

    # Ratios (safe division)
    df["digit_ratio"] = [
        _safe_ratio(n, l) for n, l in zip(df["digit_count"], df["url_length"])
    ]
    df["special_ratio"] = [
        _safe_ratio(n, l) for n, l in zip(df["special_count"], df["url_length"])
    ]

    # Entropy
    df["url_entropy"] = df["clean_url"].apply(shannon_entropy)
    df["host_entropy"] = hosts.apply(shannon_entropy)

    # Binary features
    df["has_ip"] = df["clean_url"].str.contains(r"\d+\.\d+\.\d+\.\d+").astype(int)
    df["has_at"] = df["clean_url"].str.contains("@").astype(int)
    df["has_hyphen"] = df["clean_url"].str.contains("-").astype(int)
    df["has_https"] = df["clean_url"].str.startswith("https").astype(int)

    feature_cols = [
        "url_length",
        "host_length",
        "path_length",
        "query_length",
        "digit_count",
        "special_count",
        "dot_count",
        "subdomain_count",
        "digit_ratio",
        "special_ratio",
        "url_entropy",
        "host_entropy",
        "has_ip",
        "has_at",
        "has_hyphen",
        "has_https",
    ]

    return df, feature_cols
