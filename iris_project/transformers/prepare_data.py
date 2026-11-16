import pandas as pd
from sklearn.model_selection import train_test_split

if 'transformer' not in globals():
    transformer = lambda f: f
if 'test' not in globals():
    test = lambda f: f

@transformer
def transform(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    df = df.copy()
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["target"]
    )
    train_df["split"] = "train"
    test_df["split"] = "test"
    return pd.concat([train_df, test_df], ignore_index=True)
