import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

def padded_list(x, n, target=False):
    if len(x) >= n+1:
        if target:
            return list(x[-n:])     
        return list(x[-(n+1):-1])
    else:
        if target:
            x = list(x)
        else:
            x = list(x[:-1])
        for _ in range(n-len(x)):
            x.insert(0, 0)
        return list(x)

def make_data_dict(df, n=5, test=False):
    df_group = df.sort_values(by=["t_date"]).groupby("c_id")
    t = False
    if test:
        t = True

    data_dict = {
            "c_ids": list(df_group.groups.keys()),
            "item_ids": list(df_group.item_id.apply(lambda c: padded_list(c, n, t))),
            "category": list(df_group.category.apply(lambda c: padded_list(c, n, t))),
            "price": list(df_group.price.apply(lambda c: padded_list(c, n, t))),
            "recency": list(df_group.recency.apply(lambda c: padded_list(c, n, t))),
            "month": list(df_group.month.apply(lambda c: padded_list(c, n, t))),
            "dayofweek": list(df_group.dayofweek.apply(lambda c: padded_list(c, n, t))),
            "payment": list(df_group.payment.apply(lambda c: padded_list(c, n, t))),
            "c_since": list(df_group.c_since.apply(lambda c: min(c))),
        }

    if test==False:
        data_dict["target"] = list(df_group.item_id.apply(lambda c: padded_list(c, n, True)))
    
    return data_dict

def create_batch_data(data_set, test=False, batch=128):
    features = {"item_ids": data_set["item_ids"],
                "recency" : tf.one_hot(data_set["recency"], 100),
                "month" : tf.one_hot(data_set["month"], 12),
                "dayofweek" : tf.one_hot(data_set["dayofweek"], 7),
                "category" : tf.one_hot(data_set["category"], 13),
                "payment" : tf.one_hot(data_set["payment"], 1000),
                "price" : tf.one_hot(data_set["price"], 500),
                "c_ids" : data_set["c_ids"],
                "c_since" : tf.one_hot(data_set["c_since"], 100)}
    if test:
        batch_data = tf.data.Dataset.from_tensor_slices(dict(features))
    else:
        target = data_set["target"]
        batch_data = tf.data.Dataset.from_tensor_slices((dict(features), target))
    
    batch_data = batch_data.batch(batch)

    return batch_data

def model_training(model, train_set, val_set, verbose=0):
  
    es = keras.callbacks.EarlyStopping(monitor="val_loss",
                                       mode="min",
                                       verbose=1,
                                       patience=3)
    
    hist = model.fit(train_set,
                     epochs=200,
                     verbose=verbose,
                     validation_data=val_set,
                     callbacks=[es])
    return hist

def precision_k(r, k):
    r = np.asarray(r)[:k] != 0
    return np.sum(r)/k

def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def dcg_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_k(r, k, method=0):
    dcg_max = dcg_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_k(r, k, method) / dcg_max