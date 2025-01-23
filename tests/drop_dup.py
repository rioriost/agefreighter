#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import csv

columns = [
    "CustomerID",
    "start_vertex_type",
    "Name",
    "Address",
    "Email",
    "Phone",
    "ProductID",
    "end_vertex_type",
    "Phrase",
    "SKU",
    "Price",
    "Color",
    "Size",
    "Weight",
]
df = pd.read_csv("../data/transaction/customer_product_bought.csv", usecols=columns)
df = df.drop_duplicates()
df.index = range(1, len(df) + 1)
df.to_csv(
    "../data/transaction/customer_product_bought_uniq.csv",
    index=True,
    header=True,
    quotechar='"',
    quoting=csv.QUOTE_ALL,
)
