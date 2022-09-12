# -*- coding: utf-8 -*-
"""data_preprocess.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1M000TN38FW9WYf7micE7HKiMdDw1Rm-P
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

!wget "https://github.com/ibrahimerdem/application_storage/raw/main/ecommerce_dataset.zip"
!unzip "ecommerce_dataset.zip"
data = pd.read_csv("ecommerce_dataset.csv")

df = data.copy()

df = df[df["status"]=="complete"]
df = df.iloc[:, [0, 2, 20, 3, 8, 4, 5, 17]]
df.rename(columns={"item_id":"t_id",
                   "created_at":"t_date",
                   "Customer ID":"c_id",
                   "sku":"item_name",
                   "category_name_1":"item_category",
                   "qty_ordered":"amount",
                   "Customer Since":"c_since"}, inplace=True)
df["t_date"] = df["t_date"].astype("datetime64")
df["c_since"] = df["c_since"].astype("datetime64")

END_DATE = "2018-07-01"
C_FREQ = 1
P_FREQ = 2
CAT_FREQ = 3500

df["item_category"].value_counts().describe().T

df =  df[df["t_date"]<datetime.strptime(END_DATE, "%Y-%m-%d")]

freq = pd.DataFrame(df["item_name"].value_counts())
f_items = freq[freq["item_name"]>P_FREQ].index
df = df[df["item_name"].isin(f_items)]

freq = pd.DataFrame(df["c_id"].value_counts())
f_cust = freq[freq["c_id"]>C_FREQ].index
model_df = df.loc[(df["c_id"].isin(f_cust))]

fcat = pd.DataFrame(model_df["item_category"].value_counts())
f_cat = fcat.loc[fcat["item_category"]<CAT_FREQ, ["item_category"]]
model_df.loc[model_df["item_category"].isin(f_cat.index), "item_category"] = "other"

items = model_df["item_name"].unique()
i_size = len(items)
item2code = {}
code2item = {}
for num, item in enumerate(items):
    item2code[item] = num+1
    code2item[num+1] = item

model_df["item_id"] = model_df["item_name"].map(item2code)

cats = model_df["item_category"].unique()
c_size = len(cats)
cat2code = {}
code2cat = {}
for num, cat in enumerate(cats):
    cat2code[cat] = num+1
    code2cat[num+1] = cat

model_df["category"] = model_df["item_category"].map(cat2code)
model_df = model_df.drop(columns=["item_name", "item_category"])

model_df["t_id"] = model_df["t_id"].astype("int")
model_df["c_id"] = model_df["c_id"].astype("int")
model_df["price"] = model_df["price"].astype("float")
model_df["amount"] = model_df["amount"].astype("int")

mind = model_df["t_date"].min()
maxd = model_df["t_date"].max()

print("modeling filter")
print("\ndate between:", mind.day, mind.month, mind.year,
      "-", maxd.day, maxd.month, maxd.year)
print("#transactions:", model_df["t_id"].nunique())
print("#product-categories:", model_df["category"].nunique())
print("#customers:", model_df["c_id"].nunique(),
      "#items", model_df["item_id"].nunique())

model_df.info()

model_df.to_csv("model_data_t1.csv", index=None)