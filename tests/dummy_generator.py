#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-

import csv
from faker import Faker
import random
import time

fake = Faker()

# create customers
header = ['id', 'name', 'address', 'email', 'phone']
row_cnt  = 100000000
item_cnt = 100000
names = [fake.name() for i in range(item_cnt)]
addresses = [fake.address().replace('\n',' ') for i in range(item_cnt)]
emails = [fake.email() for i in range(item_cnt)]
phones = [fake.phone_number() for i in range(item_cnt)]
customers = []
for i in range(row_cnt):
    id = f"{i + 1:09}"
    rec_no = i % item_cnt
    customers.append([id, names[rec_no], addresses[rec_no], emails[rec_no], phones[rec_no]])

with open('customers.csv', 'w') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(header)
    writer.writerows(customers)
f.close()

# create products
header = ['id', 'name', 'SKU', 'price', 'specifications']
row_cnt = 10000
products = []
for i in range(row_cnt):
    id = f"{i + 1:05}"
    name = fake.catch_phrase()
    SKU = fake.ean13()
    price = round(fake.random_number(digits=5, fix_len=True) / 100, 2)
    specifications = f"Color: {fake.color_name()}; Size: {fake.random_element(elements=('S', 'M', 'L', 'XL'))}; Weight: {fake.random_int(min=100, max=1000)}g"
    products.append([id, name, SKU, price, specifications])

with open('products.csv', 'w') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(header)
    writer.writerows(products)
f.close()

# create boughts (edges)
header = ['start_id', 'start_vertex_type', 'end_id', 'end_vertex_type']
row_cnt = len(customers)
boughts = []
for i in range(row_cnt):
    start_id = customers[i][0]
    start_vertex_type = 'Customer'
    [boughts.append([start_id, start_vertex_type, fake.random_int(min=1, max=len(products)), 'Product'])  for _ in range(random.randrange(1, 5))]

with open('boughts.csv', 'w') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(header)
    writer.writerows(boughts)
f.close()