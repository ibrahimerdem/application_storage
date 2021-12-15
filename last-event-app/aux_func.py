import pandas as pd
import tensorflow as tf
from tensorflow import keras

def split_input_target(sequence):
    sequence = list(sequence)
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return {"input_seq": input_text, "output_seq": target_text}

def create_sequences(df):
    df["item_code"] = df["item_code"].astype("string")
    df["recency"] = df["recency"].astype("string")
    df["payment"] = df["payment"].astype("string")
    df["category_code"] = df["category_code"].astype("string")
    df["method_code"] = df["method_code"].astype("string")
    df["month"] = df["month"].astype("string")
    df["dayofweek"] = df["dayofweek"].astype("string")
    g_df = df.groupby("c_id", as_index=False).agg(
          {"item_code": lambda x: ", ".join(x),
          "recency": lambda y: ", ".join(y),
          "payment": lambda z: ", ".join(z),
          "category_code": lambda t: ", ".join(t),
          "method_code": lambda u: ", ".join(u),
          "month": lambda v: ", ".join(v),
          "dayofweek": lambda p: ", ".join(p)}
          )
    g_df["item_code"] = g_df["item_code"].apply(
      lambda x: x.split(", ")
      )
    g_df["recency"] = g_df["recency"].apply(
      lambda y: y.split(", ")
      )
    g_df["payment"] = g_df["payment"].apply(
      lambda z: z.split(", ")
      )
    g_df["category_code"] = g_df["category_code"].apply(
      lambda t: t.split(", ")
      )
    g_df["method_code"] = g_df["method_code"].apply(
      lambda u: u.split(", ")
      )
    g_df["month"] = g_df["month"].apply(
      lambda v: v.split(", ")
      )
    g_df["dayofweek"] = g_df["dayofweek"].apply(
      lambda p: p.split(", ")
      )
    return g_df

def create_splits(g_df):
    s_df = g_df.merge(g_df["item_code"].apply(
      lambda w: pd.Series(split_input_target(w))
      ), left_index=True, right_index=True)
    s_df["recency"] = s_df["recency"].apply(lambda p: list(p)[:-1])
    s_df["payment"] = s_df["payment"].apply(lambda q: list(q)[:-1])
    s_df["category_code"] = s_df["category_code"].apply(lambda r: list(r)[:-1])
    s_df["method_code"] = s_df["method_code"].apply(lambda s: list(s)[:-1])
    s_df["month"] = s_df["month"].apply(lambda t: list(t)[:-1])
    s_df["dayofweek"] = s_df["dayofweek"].apply(lambda u: list(u)[:-1])
    return s_df

def make_padding(df, n_padded):
    input_seq_padded = keras.preprocessing.sequence.pad_sequences(
        df["input_seq"].values, 
        maxlen=n_padded,
        padding="pre",
        value=0.0
    )
    recency_seq_padded = keras.preprocessing.sequence.pad_sequences(
        df["recency"].values,
        maxlen=n_padded,
        padding="pre",
        value=0.0
    )
    payment_seq_padded = keras.preprocessing.sequence.pad_sequences(
        df["payment"].values,
        maxlen=n_padded,
        padding="pre",
        value=0.0
    )
    category_seq_padded = keras.preprocessing.sequence.pad_sequences(
        df["category_code"].values,
        maxlen=n_padded,
        padding="pre",
        value=0.0
    )
    method_seq_padded = keras.preprocessing.sequence.pad_sequences(
        df["method_code"].values,
        maxlen=n_padded,
        padding="pre",
        value=0.0
    )
    month_seq_padded = keras.preprocessing.sequence.pad_sequences(
        df["month"].values,
        maxlen=n_padded,
        padding="pre",
        value=0.0
    )
    day_seq_padded = keras.preprocessing.sequence.pad_sequences(
        df["dayofweek"].values,
        maxlen=n_padded,
        padding="pre",
        value=0.0
    )
    if len(df.columns[df.columns.str.contains("output_seq")]) > 0:
        output_seq_padded = keras.preprocessing.sequence.pad_sequences(
            df["output_seq"].values,
            maxlen=n_padded,
            padding="pre",
            value=0.0
        )
    else:
        output_seq_padded = []
    return [input_seq_padded,
            recency_seq_padded,
            payment_seq_padded,
            category_seq_padded,
            method_seq_padded,
            month_seq_padded,
            day_seq_padded,
            output_seq_padded]