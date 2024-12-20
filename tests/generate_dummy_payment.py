#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
import csv
from faker import Faker
import random

fake = Faker()

print("Creating payments...")
# create customers
payments_header = ["name", "address", "email", "phone"]
row_cnt = 70000000
item_cnt = 100000
names = [fake.name() for i in range(item_cnt)]
addresses = [fake.address().replace("\n", " ") for i in range(item_cnt)]
emails = [fake.email() for i in range(item_cnt)]
phones = [fake.phone_number() for i in range(item_cnt)]
payments = [
    {
        "CustomerID": f"{i + 1:09}",
        "properties": {
            "name": names[i % item_cnt],
            "address": addresses[i % item_cnt],
            "email": emails[i % item_cnt],
            "phone": phones[i % item_cnt],
        },
    }
    for i in range(row_cnt)
]

print("Creating users...")
# create products
users_header = ["desc", "SKU", "price", "Color", "Size", "Weight"]
row_cnt = 40000000
item_cnt = 100000
names = [fake.name() for i in range(item_cnt)]
addresses = [fake.address().replace("\n", " ") for i in range(item_cnt)]
emails = [fake.email() for i in range(item_cnt)]
phones = [fake.phone_number() for i in range(item_cnt)]
users = [
    {
        "CustomerID": f"{i + 1:09}",
        "properties": {
            "name": names[i % item_cnt],
            "address": addresses[i % item_cnt],
            "email": emails[i % item_cnt],
            "phone": phones[i % item_cnt],
        },
    }
    for i in range(row_cnt)
]

print("Creating cookies...")
# create products
cookies_header = ["desc", "SKU", "price", "Color", "Size", "Weight"]
row_cnt = 30000000
item_cnt = 100000
names = [fake.name() for i in range(item_cnt)]
addresses = [fake.address().replace("\n", " ") for i in range(item_cnt)]
emails = [fake.email() for i in range(item_cnt)]
phones = [fake.phone_number() for i in range(item_cnt)]
cookies = [
    {
        "CustomerID": f"{i + 1:09}",
        "properties": {
            "name": names[i % item_cnt],
            "address": addresses[i % item_cnt],
            "email": emails[i % item_cnt],
            "phone": phones[i % item_cnt],
        },
    }
    for i in range(row_cnt)
]

print("Creating IPs...")
# create products
ips_header = ["desc", "SKU", "price", "Color", "Size", "Weight"]
row_cnt = 20000000
item_cnt = 100000
names = [fake.name() for i in range(item_cnt)]
addresses = [fake.address().replace("\n", " ") for i in range(item_cnt)]
emails = [fake.email() for i in range(item_cnt)]
phones = [fake.phone_number() for i in range(item_cnt)]
ips = [
    {
        "CustomerID": f"{i + 1:09}",
        "properties": {
            "name": names[i % item_cnt],
            "address": addresses[i % item_cnt],
            "email": emails[i % item_cnt],
            "phone": phones[i % item_cnt],
        },
    }
    for i in range(row_cnt)
]

print("Creating Encrypted Addresses...")
# create products
addresses_header = ["desc", "SKU", "price", "Color", "Size", "Weight"]
row_cnt = 20000000
item_cnt = 100000
names = [fake.name() for i in range(item_cnt)]
addresses = [fake.address().replace("\n", " ") for i in range(item_cnt)]
emails = [fake.email() for i in range(item_cnt)]
phones = [fake.phone_number() for i in range(item_cnt)]
addresses = [
    {
        "CustomerID": f"{i + 1:09}",
        "properties": {
            "name": names[i % item_cnt],
            "address": addresses[i % item_cnt],
            "email": emails[i % item_cnt],
            "phone": phones[i % item_cnt],
        },
    }
    for i in range(row_cnt)
]

print("Creating Credit Cards...")
# create products
cards_header = ["desc", "SKU", "price", "Color", "Size", "Weight"]
row_cnt = 10000000
item_cnt = 100000
names = [fake.name() for i in range(item_cnt)]
addresses = [fake.address().replace("\n", " ") for i in range(item_cnt)]
emails = [fake.email() for i in range(item_cnt)]
phones = [fake.phone_number() for i in range(item_cnt)]
cards = [
    {
        "CustomerID": f"{i + 1:09}",
        "properties": {
            "name": names[i % item_cnt],
            "address": addresses[i % item_cnt],
            "email": emails[i % item_cnt],
            "phone": phones[i % item_cnt],
        },
    }
    for i in range(row_cnt)
]

print("Creating Emails...")
# create products
emails_header = ["desc", "SKU", "price", "Color", "Size", "Weight"]
row_cnt = 10000000
item_cnt = 100000
names = [fake.name() for i in range(item_cnt)]
addresses = [fake.address().replace("\n", " ") for i in range(item_cnt)]
emails = [fake.email() for i in range(item_cnt)]
phones = [fake.phone_number() for i in range(item_cnt)]
emails = [
    {
        "CustomerID": f"{i + 1:09}",
        "properties": {
            "name": names[i % item_cnt],
            "address": addresses[i % item_cnt],
            "email": emails[i % item_cnt],
            "phone": phones[i % item_cnt],
        },
    }
    for i in range(row_cnt)
]

print("Creating Phones...")
# create products
phones_header = ["desc", "SKU", "price", "Color", "Size", "Weight"]
row_cnt = 10000000
item_cnt = 100000
names = [fake.name() for i in range(item_cnt)]
addresses = [fake.address().replace("\n", " ") for i in range(item_cnt)]
emails = [fake.email() for i in range(item_cnt)]
phones = [fake.phone_number() for i in range(item_cnt)]
phones = [
    {
        "CustomerID": f"{i + 1:09}",
        "properties": {
            "name": names[i % item_cnt],
            "address": addresses[i % item_cnt],
            "email": emails[i % item_cnt],
            "phone": phones[i % item_cnt],
        },
    }
    for i in range(row_cnt)
]

print("Creating Crypto Money Addresses...")
# create products
crypto_coins_header = ["desc", "SKU", "price", "Color", "Size", "Weight"]
row_cnt = 10000000
item_cnt = 100000
names = [fake.name() for i in range(item_cnt)]
addresses = [fake.address().replace("\n", " ") for i in range(item_cnt)]
emails = [fake.email() for i in range(item_cnt)]
phones = [fake.phone_number() for i in range(item_cnt)]
crypto_coins = [
    {
        "CustomerID": f"{i + 1:09}",
        "properties": {
            "name": names[i % item_cnt],
            "address": addresses[i % item_cnt],
            "email": emails[i % item_cnt],
            "phone": phones[i % item_cnt],
        },
    }
    for i in range(row_cnt)
]


print("Creating transactions...")
# create boughts (edges)
header = (
    ["start_id", "start_vertex_type", "end_id", "end_vertex_type"]
    + cst_header
    + prd_header
)
with open("transactions.csv", "w", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(header)
    row_cnt = len(customers)
    for i in range(0, row_cnt, 100000):
        boughts = []
        for customer in customers[i : i + 10000]:
            num_boughts = random.randrange(1, 5)
            for _ in range(num_boughts):
                product_id = fake.random_int(min=1, max=len(products) - 1)
                boughts.append(
                    [
                        customer["CustomerID"],
                        "Customer",
                        product_id,
                        "Product",
                        *[str(v) for v in customer["properties"].values()],
                        *[str(v) for v in products[product_id]["properties"].values()],
                    ]
                )
        writer.writerows(boughts)
f.close()
