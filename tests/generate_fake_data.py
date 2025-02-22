#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import asyncio

import aiofiles
from faker import Faker

fake = Faker()


class Node:
    def __init__(self, id: str = "", properties: dict = {}):
        self.id = id
        self.properties = properties


class Edge:
    def __init__(
        self,
        id: str = "",
        properties: dict = {},
    ):
        self.id = id
        self.properties = properties


# nodes
class AirPort(Node):
    def __init__(self, id: str = ""):
        city = fake.city()
        super().__init__(
            id=id,
            properties={
                "Name": f"{city} Airport",
                "City": city,
                "Country": fake.country(),
                "IATA": fake.unique.bothify(text="???").upper(),
                "ICAO": fake.unique.bothify(text="????").upper(),
                "Latitude": fake.latitude(),
                "Longitude": fake.longitude(),
                "Altitude": fake.random_int(min=0, max=1000),
                "Timezone": fake.timezone(),
                "DST": random.choice(["E", "A", "S", "O", "Z", "N", "U"]),
                "Tz": fake.timezone(),
            },
        )


class Customer(Node):
    def __init__(self, id: str = ""):
        super().__init__(
            id=id,
            properties={
                "Name": fake.name(),
                "Address": fake.address().replace("\n", " "),
                "Email": fake.email(),
                "Phone": fake.phone_number(),
            },
        )


class Product(Node):
    def __init__(self, id: str = ""):
        super().__init__(
            id=id,
            properties={
                "Phrase": fake.catch_phrase(),
                "SKU": fake.ean13(),
                "Price": round(fake.random_number(digits=5, fix_len=True) / 100, 2),
                "Color": fake.color_name(),
                "Size": fake.random_element(elements=("S", "M", "L", "XL")),
                "Weight": fake.random_int(min=100, max=1000),
            },
        )


class Manufacturer(Node):
    def __init__(self, id: str = ""):
        super().__init__(
            id=id,
            properties={
                "Name": fake.company(),
                "Address": fake.address(),
                "Phone": fake.phone_number(),
                "Email": fake.email(),
                "Website": fake.url(),
            },
        )


class Country(Node):
    def __init__(self, id: str = ""):
        super().__init__(
            id=id,
            properties={
                "Name": fake.country(),
                "Capital": fake.city(),
                "Population": fake.random_int(min=1000000, max=1000000000),
                "ISO": fake.unique.bothify(text="??").upper(),
                "TLD": fake.unique.bothify(text="???").lower(),
                "FlagURL": fake.image_url(),
            },
        )


class City(Node):
    def __init__(self, id: str = ""):
        super().__init__(
            id=id,
            properties={
                "Name": fake.city(),
                "Latitude": fake.latitude(),
                "Longitude": fake.longitude(),
            },
        )


class BitcoinAddress(Node):
    def __init__(self, id: str = ""):
        dt = fake.date_time_this_year(before_now=True, after_now=False, tzinfo=None)
        super().__init__(
            id=id,
            properties={
                "available_since": dt,
                "inserted_at": dt,
                "address": random.choice(["cryptoaddress:", "bitcoinaddress:"])
                + generate_base58_dummy_data(16),
                "schema_version": "1",
            },
        )


class Cookie(Node):
    def __init__(self, id: str = ""):
        dt = fake.date_time_this_year(before_now=True, after_now=False, tzinfo=None)
        super().__init__(
            id=id,
            properties={
                "available_since": dt,
                "inserted_at": dt,
                "uaid": fake.uuid4(),
                "schema_version": "1",
            },
        )


class IP(Node):
    def __init__(self, id: str = ""):
        dt = fake.date_time_this_year(before_now=True, after_now=False, tzinfo=None)
        super().__init__(
            id=id,
            properties={
                "available_since": dt,
                "inserted_at": dt,
                "address": fake.ipv4(),
                "schema_version": "1",
            },
        )


class Phone(Node):
    def __init__(self, id: str = ""):
        dt = fake.date_time_this_year(before_now=True, after_now=False, tzinfo=None)
        super().__init__(
            id=id,
            properties={
                "available_since": dt,
                "inserted_at": dt,
                "address": fake.ipv4(),
                "schema_version": "1",
            },
        )


class Email(Node):
    def __init__(self, id: str = ""):
        dt = fake.date_time_this_year(before_now=True, after_now=False, tzinfo=None)
        email = fake.email()
        handle, domain = email.split("@")
        super().__init__(
            id=id,
            properties={
                "available_since": dt,
                "inserted_at": dt,
                "email": email,
                "domain": domain,
                "handle": handle,
                "schema_version": "1",
            },
        )


class Payment(Node):
    def __init__(self, id: str = ""):
        dt = fake.date_time_this_year(before_now=True, after_now=False, tzinfo=None)
        super().__init__(
            id=id,
            properties={
                "available_since": dt,
                "inserted_at": dt,
                "payment_id": fake.uuid4(),
                "schema_version": "1",
            },
        )


class PartnerEndUser(Node):
    def __init__(self, id: str = ""):
        dt = fake.date_time_this_year(before_now=True, after_now=False, tzinfo=None)
        super().__init__(
            id=id,
            properties={
                "available_since": dt,
                "inserted_at": dt,
                "partner_end_user_id": fake.uuid4(),
                "schema_version": "1",
            },
        )


class CreditCard(Node):
    def __init__(self, id: str = ""):
        dt = fake.date_time_this_year(before_now=True, after_now=False, tzinfo=None)
        super().__init__(
            id=id,
            properties={
                "available_since": dt,
                "inserted_at": dt,
                "expiry_month": fake.month(),
                "expiry_year": fake.year(),
                "masked_number": fake.credit_card_number(card_type=None),
                "creditcard_identifier": fake.uuid4(),
                "schema_version": "1",
            },
        )


class CryptoAddress(Node):
    def __init__(self, id: str = ""):
        dt = fake.date_time_this_year(before_now=True, after_now=False, tzinfo=None)
        super().__init__(
            id=id,
            properties={
                "available_since": dt,
                "inserted_at": dt,
                "address": generate_base58_dummy_data(16),
                "currency": random.choice(["BTC", "ETH", "LTC", "XRP"]),
                "full_address": generate_base58_dummy_data(32),
                "schema_version": "1",
                "tag": generate_base58_dummy_data(4),
            },
        )


# edges
class AirRoute(Edge):
    def __init__(
        self,
        id: str = "",
        start_id: str = "",
        start_vertex_type: str = "",
        start_props: dict = {},
        end_id: str = "",
        end_vertex_type: str = "",
        end_props: dict = {},
    ):
        super().__init__(
            id=id,
            properties={
                "start_id": start_id,
                "start_vertex_type": start_vertex_type,
                **{f"start_{k}": v for k, v in start_props.items()},
                "end_id": end_id,
                "end_vertex_type": end_vertex_type,
                **{f"end_{k}": v for k, v in end_props.items()},
                "distance": fake.random_int(min=100, max=16000),
            },
        )


class Bought(Edge):
    def __init__(
        self,
        id: str = "",
        start_id: str = "",
        start_vertex_type: str = "",
        start_props: dict = {},
        end_id: str = "",
        end_vertex_type: str = "",
        end_props: dict = {},
    ):
        super().__init__(
            id=id,
            properties={
                "start_id": start_id,
                "start_vertex_type": start_vertex_type,
                **{f"{k}": v for k, v in start_props.items()},
                "end_id": end_id,
                "end_vertex_type": end_vertex_type,
                **{f"{k}": v for k, v in end_props.items()},
            },
        )


class Has(Edge):
    def __init__(
        self,
        id: str = "",
        start_id: str = "",
        start_vertex_type: str = "",
        start_props: dict = {},
        end_id: str = "",
        end_vertex_type: str = "",
        end_props: dict = {},
    ):
        super().__init__(
            id=id,
            properties={
                "start_id": start_id,
                "start_vertex_type": start_vertex_type,
                **{f"start_{k}": v for k, v in start_props.items()},
                "end_id": end_id,
                "end_vertex_type": end_vertex_type,
                **{f"end_{k}": v for k, v in end_props.items()},
                "since": fake.date_time(),
            },
        )


class PerformedBy(Edge):
    def __init__(
        self,
        id: str = "",
        start_id: str = "",
        start_vertex_type: str = "",
        end_id: str = "",
        end_vertex_type: str = "",
    ):
        dt = fake.date_time_this_year(before_now=True, after_now=False, tzinfo=None)
        super().__init__(
            id=id,
            properties={
                "start_id": start_id,
                "start_vertex_type": start_vertex_type,
                "end_id": end_id,
                "end_vertex_type": end_vertex_type,
                "available_since": dt,
                "inserted_at": dt,
            },
        )


class UsedBy(Edge):
    def __init__(
        self,
        id: str = "",
        start_id: str = "",
        start_vertex_type: str = "",
        end_id: str = "",
        end_vertex_type: str = "",
    ):
        dt = fake.date_time_this_year(before_now=True, after_now=False, tzinfo=None)
        super().__init__(
            id=id,
            properties={
                "start_id": start_id,
                "start_vertex_type": start_vertex_type,
                "end_id": end_id,
                "end_vertex_type": end_vertex_type,
                "available_since": dt,
                "inserted_at": dt,
                "schema_version": "1",
            },
        )


class UsedIn(Edge):
    def __init__(
        self,
        id: str = "",
        start_id: str = "",
        start_vertex_type: str = "",
        end_id: str = "",
        end_vertex_type: str = "",
    ):
        dt = fake.date_time_this_year(before_now=True, after_now=False, tzinfo=None)
        super().__init__(
            id=id,
            properties={
                "start_id": start_id,
                "start_vertex_type": start_vertex_type,
                "end_id": end_id,
                "end_vertex_type": end_vertex_type,
                "available_since": dt,
                "inserted_at": dt,
                "schema_version": "1",
            },
        )


class Produce(Edge):
    def __init__(
        self,
        id: str = "",
        start_id: str = "",
        start_vertex_type: str = "",
        start_props: dict = {},
        end_id: str = "",
        end_vertex_type: str = "",
        end_props: dict = {},
    ):
        super().__init__(
            id=id,
            properties={
                "start_id": start_id,
                "start_vertex_type": start_vertex_type,
                **{f"{k}": v for k, v in start_props.items()},
                "end_id": end_id,
                "end_vertex_type": end_vertex_type,
                **{f"{k}": v for k, v in end_props.items()},
            },
        )


def generate_base58_dummy_data(length: int = 0) -> str:
    """
    Generate a random base58 string of a given length.

    Args:
        length (int): The length of the base58 string to generate.

    Returns:
        str: The generated base58 string.
    """
    base58_chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    return "".join(random.choice(base58_chars) for _ in range(length))


async def put_csv(
    data_dir: str = "../data/", file_name: str = "dummy.csv", data: list = []
) -> None:
    """
    Write a list of objects to a CSV file.

    Args:
        data_dir (str): The directory where the CSV file will be saved.
        file_name (str): The name of the CSV file.
        data (list): A list of objects to be written to the CSV file.

    Returns:
        None
    """
    fpath = os.path.abspath(
        os.path.expanduser(os.path.dirname(__file__) + "/" + data_dir)
    )
    os.makedirs(fpath, exist_ok=True)
    fname = f"{fpath}/{file_name.lower()}.csv"
    headers = ['"id"'] + [f'"{k}"' for k in data[0].properties]
    async with aiofiles.open(fname, "w") as f:
        await f.write(",".join(headers) + "\n")
        await f.write(
            "\n".join(
                [
                    ",".join([f'"{d.id}"'] + [f'"{v}"' for v in d.properties.values()])
                    for d in data
                ]
            )
        )


async def generate_complete_data(
    edge_cls_name: str = "",
    edge_props: dict = {},
    nodes: dict = {},
    data_dir: str = "",
) -> None:
    """
    Generate a list of objects of a given class and write them to a CSV file.

    Args:
        edge_cls_name (str): The name of the class of the edge.
        edge_count (int): The number of edges to generate.
        nodes (dict): A dictionary containing the names of the classes of the nodes and the number of nodes to generate.
        data_dir (str): The directory where the CSV file will be saved.

    Returns:
        None
    """
    print(f"Generating {edge_cls_name}: {edge_props['count']}...")

    start_node_data = [
        globals()[edge_props["start"]](id=i + 1)
        for i in range(nodes[edge_props["start"]])
    ]
    end_node_data = [
        globals()[edge_props["end"]](id=i + 1) for i in range(nodes[edge_props["end"]])
    ]

    data = []
    for i in range(edge_props["count"]):
        start = random.choice(start_node_data)
        end = random.choice(end_node_data)
        data.append(
            globals()[edge_cls_name](
                id=i + 1,
                start_id=start.id,
                start_vertex_type=edge_props["start"],
                start_props=start.properties,
                end_id=end.id,
                end_vertex_type=edge_props["end"],
                end_props=end.properties,
            )
        )
    await put_csv(
        data_dir=data_dir,
        file_name=f"{edge_props['start']}_{edge_props['end']}_{edge_cls_name}",
        data=data,
    )


async def generate_nodes(
    cls_name: str = "", count: int = 0, data_dir: str = ""
) -> tuple:
    """
    Generate a list of objects of a given class and write them to a CSV file.

    Args:
        cls_name (str): The name of the class to be generated.
        count (int): The number of objects to generate.
        data_dir (str): The directory where the CSV file will be saved.

    Returns:
        tuple: A tuple containing the class name and the list of generated objects.
    """
    print(f"Generating {cls_name}: {count}...")
    data = [globals()[cls_name](id=i + 1) for i in range(count)]
    await put_csv(data_dir=data_dir, file_name=cls_name, data=data)
    return cls_name, data


async def generate_edges(
    cls_name: str = "", prop_list: list = [], nodes_data: list = [], data_dir: str = ""
) -> None:
    """
    Generate a list of objects of a given class and write them to a CSV file.

    Args:
        cls_name (str): The name of the class to be generated.
        prop_list (list): A list of dictionaries containing the properties of the class.
        nodes_data (list): A list of dictionaries containing the data of the nodes.
        data_dir (str): The directory where the CSV file will be saved.

    Returns:
        None
    """
    for props in prop_list:
        print(
            f"Generating {cls_name} {props['start']} - {props['end']}: {props['count']}..."
        )
        start_vertex_type = props["start"]
        end_vertex_type = props["end"]
        data = [
            globals()[cls_name](
                id=i + 1,
                start_id=random.choice(nodes_data[props["start"]]).id,
                start_vertex_type=start_vertex_type,
                end_id=random.choice(nodes_data[props["end"]]).id,
                end_vertex_type=end_vertex_type,
            )
            for i in range(props["count"])
        ]
        await put_csv(
            data_dir=data_dir,
            file_name=f"{cls_name}_{props['start']}_{props['end']}",
            data=data,
        )


async def main() -> None:
    SINGLE_SOURCE = 1
    MULTI_SOURCE = 2

    TEST_PATTERN = 4

    # transaction, large data for AzureStorageFreighter, multiple types of nodes and edges
    if TEST_PATTERN == 1:
        source_type = SINGLE_SOURCE
        DATA_DIR = "../data/transaction"
        nodes = {"Customer": 10000, "Product": 1000}
        edges = {"Bought": {"count": 20050, "start": "Customer", "end": "Product"}}
    # airroute, small data for CSVFreighter, single type of nodes and edges
    elif TEST_PATTERN == 2:
        source_type = MULTI_SOURCE
        DATA_DIR = "../data/airroute"
        nodes = {"AirPort": 3500}
        edges = {"AirRoute": [{"count": 20000, "start": "AirPort", "end": "AirPort"}]}
    # countries, small data for MultiCSVFreighter, multiple types of nodes and edges
    elif TEST_PATTERN == 3:
        source_type = MULTI_SOURCE
        DATA_DIR = "../data/countries"
        nodes = {"Country": 200, "City": 10000}
        edges = {"Has": [{"count": 10000, "start": "Country", "end": "City"}]}
    # payment, large data for AzureStorageFreighter
    elif TEST_PATTERN == 4 or TEST_PATTERN == 5:
        source_type = MULTI_SOURCE
        if TEST_PATTERN == 4:
            DIVIDER = 10000
            DATA_DIR = "../data/payment_small"
        elif TEST_PATTERN == 5:
            DIVIDER = 10
            DATA_DIR = "../data/payment_large"

        nodes = {
            "BitcoinAddress": 9000000 // DIVIDER,
            "Cookie": 27000000 // DIVIDER,
            "IP": 22000000 // DIVIDER,
            "Phone": 9600000 // DIVIDER,
            "Email": 9600000 // DIVIDER,
            "Payment": 70000000 // DIVIDER,
            "CreditCard": 12000000 // DIVIDER,
            "PartnerEndUser": 40000000 // DIVIDER,
            "CryptoAddress": 16000000 // DIVIDER,
        }

        edges = {
            "UsedIn": [
                {"count": 60000000 // DIVIDER, "start": "Cookie", "end": "Payment"},
                {"count": 60000000 // DIVIDER, "start": "Email", "end": "Payment"},
                {
                    "count": 60000000 // DIVIDER,
                    "start": "CryptoAddress",
                    "end": "Payment",
                },
                {"count": 60000000 // DIVIDER, "start": "Phone", "end": "Payment"},
                {"count": 60000000 // DIVIDER, "start": "CreditCard", "end": "Payment"},
            ],
            "PerformedBy": [
                {"count": 10000000 // DIVIDER, "start": "Cookie", "end": "Payment"},
                {"count": 10000000 // DIVIDER, "start": "Email", "end": "Payment"},
                {
                    "count": 10000000 // DIVIDER,
                    "start": "CryptoAddress",
                    "end": "Payment",
                },
                {"count": 10000000 // DIVIDER, "start": "Phone", "end": "Payment"},
                {"count": 10000000 // DIVIDER, "start": "CreditCard", "end": "Payment"},
            ],
            "UsedBy": [
                {"count": 80000000 // DIVIDER, "start": "Cookie", "end": "Payment"},
                {"count": 80000000 // DIVIDER, "start": "Email", "end": "Payment"},
                {
                    "count": 80000000 // DIVIDER,
                    "start": "CryptoAddress",
                    "end": "Payment",
                },
                {"count": 80000000 // DIVIDER, "start": "Phone", "end": "Payment"},
                {"count": 80000000 // DIVIDER, "start": "CreditCard", "end": "Payment"},
            ],
        }

    if source_type == SINGLE_SOURCE:
        for edge_cls_name, prop_list in edges.items():
            await generate_complete_data(
                edge_cls_name=edge_cls_name,
                edge_props=prop_list,
                nodes=nodes,
                data_dir=DATA_DIR,
            )
    elif source_type == MULTI_SOURCE:
        node_tasks = [
            generate_nodes(cls_name, count, DATA_DIR)
            for cls_name, count in nodes.items()
        ]
        nodes_data = dict(await asyncio.gather(*node_tasks))

        edge_tasks = [
            generate_edges(cls_name, prop_list, nodes_data, DATA_DIR)
            for cls_name, prop_list in edges.items()
        ]
        await asyncio.gather(*edge_tasks)


if __name__ == "__main__":
    asyncio.run(main())
