#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from datetime import datetime
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Type

import aiofiles
from faker import Faker

# Configure logging; default to INFO (overridable by the --debug flag)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

fake = Faker()


def get_timestamp() -> Dict[str, Any]:
    """Return a dictionary with a common timestamp for nodes/edges."""
    dt = fake.date_time_this_year(before_now=True, after_now=False, tzinfo=None)
    return {"available_since": dt, "inserted_at": dt}


def prefixed_props(prefix: str, props: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a dictionary with each key prefixed if props is provided."""
    if props is None:
        return {}
    return {f"{prefix}{k}": v for k, v in props.items()}


class Node:
    def __init__(self, id: str, properties: Optional[Dict[str, Any]]) -> None:
        if properties is None:
            properties = {}
        self.id = id
        self.properties = properties


class Edge:
    def __init__(self, id: str, properties: Optional[Dict[str, Any]]) -> None:
        if properties is None:
            properties = {}
        self.id = id
        self.properties = properties


# Node Classes
class AirPort(Node):
    def __init__(self, id: str) -> None:
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
    def __init__(self, id: str) -> None:
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
    def __init__(self, id: str) -> None:
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


class Person(Node):
    def __init__(self, id: str) -> None:
        super().__init__(
            id=id,
            properties={"Name": fake.name()},
        )


class Manufacturer(Node):
    def __init__(self, id: str) -> None:
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
    def __init__(self, id: str) -> None:
        super().__init__(
            id=id,
            properties={
                "Name": fake.country(),
                "Capital": fake.city(),
                "Population": fake.random_int(min=1_000_000, max=1_000_000_000),
                "ISO": fake.unique.bothify(text="??").upper(),
                "TLD": fake.unique.bothify(text="???").lower(),
                "FlagURL": fake.image_url(),
            },
        )


class City(Node):
    def __init__(self, id: str) -> None:
        super().__init__(
            id=id,
            properties={
                "Name": fake.city(),
                "Latitude": fake.latitude(),
                "Longitude": fake.longitude(),
            },
        )


class BitcoinAddress(Node):
    def __init__(self, id: str) -> None:
        base_props = get_timestamp()
        extra = {
            "address": random.choice(["cryptoaddress:", "bitcoinaddress:"])
            + generate_base58_dummy_data(16),
            "schema_version": "1",
        }
        base_props.update(extra)
        super().__init__(id=id, properties=base_props)


class Cookie(Node):
    def __init__(self, id: str) -> None:
        base_props = get_timestamp()
        base_props.update(
            {
                "uaid": fake.uuid4(),
                "schema_version": "1",
            }
        )
        super().__init__(id=id, properties=base_props)


class IP(Node):
    def __init__(self, id: str) -> None:
        base_props = get_timestamp()
        base_props.update(
            {
                "address": fake.ipv4(),
                "schema_version": "1",
            }
        )
        super().__init__(id=id, properties=base_props)


class Phone(Node):
    def __init__(self, id: str) -> None:
        base_props = get_timestamp()
        base_props.update(
            {
                "address": fake.ipv4(),
                "schema_version": "1",
            }
        )
        super().__init__(id=id, properties=base_props)


class Email(Node):
    def __init__(self, id: str) -> None:
        base_props = get_timestamp()
        email = fake.email()
        handle, domain = email.split("@")
        base_props.update(
            {
                "email": email,
                "domain": domain,
                "handle": handle,
                "schema_version": "1",
            }
        )
        super().__init__(id=id, properties=base_props)


class Payment(Node):
    def __init__(self, id: str) -> None:
        base_props = get_timestamp()
        base_props.update(
            {
                "payment_id": fake.uuid4(),
                "schema_version": "1",
            }
        )
        super().__init__(id=id, properties=base_props)


class PartnerEndUser(Node):
    def __init__(self, id: str) -> None:
        base_props = get_timestamp()
        base_props.update(
            {
                "partner_end_user_id": fake.uuid4(),
                "schema_version": "1",
            }
        )
        super().__init__(id=id, properties=base_props)


class CreditCard(Node):
    def __init__(self, id: str) -> None:
        base_props = get_timestamp()
        base_props.update(
            {
                "expiry_month": fake.month(),
                "expiry_year": fake.year(),
                "masked_number": fake.credit_card_number(card_type=None),
                "creditcard_identifier": fake.uuid4(),
                "schema_version": "1",
            }
        )
        super().__init__(id=id, properties=base_props)


class CryptoAddress(Node):
    def __init__(self, id: str) -> None:
        base_props = get_timestamp()
        base_props.update(
            {
                "address": generate_base58_dummy_data(16),
                "currency": random.choice(["BTC", "ETH", "LTC", "XRP"]),
                "full_address": generate_base58_dummy_data(32),
                "schema_version": "1",
                "tag": generate_base58_dummy_data(4),
            }
        )
        super().__init__(id=id, properties=base_props)


# Edge Classes
class AirRoute(Edge):
    def __init__(
        self,
        id: str,
        start_id: str,
        start_vertex_type: str,
        start_props: Optional[Dict[str, Any]],
        end_id: str,
        end_vertex_type: str,
        end_props: Optional[Dict[str, Any]],
    ) -> None:
        properties = {
            "start_id": start_id,
            "start_vertex_type": start_vertex_type,
            **prefixed_props("start_", start_props),
            "end_id": end_id,
            "end_vertex_type": end_vertex_type,
            **prefixed_props("end_", end_props),
            "distance": fake.random_int(min=100, max=16000),
        }
        super().__init__(id=id, properties=properties)


class Bought(Edge):
    def __init__(
        self,
        id: str,
        start_id: str,
        start_vertex_type: str,
        start_props: Optional[Dict[str, Any]],
        end_id: str,
        end_vertex_type: str,
        end_props: Optional[Dict[str, Any]],
    ) -> None:
        properties = {
            "start_id": start_id,
            "start_vertex_type": start_vertex_type,
            **(start_props or {}),
            "end_id": end_id,
            "end_vertex_type": end_vertex_type,
            **(end_props or {}),
        }
        super().__init__(id=id, properties=properties)


class Has(Edge):
    def __init__(
        self,
        id: str,
        start_id: str,
        start_vertex_type: str,
        start_props: Optional[Dict[str, Any]],
        end_id: str,
        end_vertex_type: str,
        end_props: Optional[Dict[str, Any]],
    ) -> None:
        properties = {
            "start_id": start_id,
            "start_vertex_type": start_vertex_type,
            **prefixed_props("start_", start_props),
            "end_id": end_id,
            "end_vertex_type": end_vertex_type,
            **prefixed_props("end_", end_props),
            "since": fake.date_time(),
        }
        super().__init__(id=id, properties=properties)


class PerformedBy(Edge):
    def __init__(
        self,
        id: str,
        start_id: str,
        start_vertex_type: str,
        start_props: Optional[Dict[str, Any]],
        end_id: str,
        end_vertex_type: str,
        end_props: Optional[Dict[str, Any]],
    ) -> None:
        properties = {
            "start_id": start_id,
            "start_vertex_type": start_vertex_type,
            "end_id": end_id,
            "end_vertex_type": end_vertex_type,
            **get_timestamp(),
        }
        super().__init__(id=id, properties=properties)


class UsedBy(Edge):
    def __init__(
        self,
        id: str,
        start_id: str,
        start_vertex_type: str,
        start_props: Optional[Dict[str, Any]],
        end_id: str,
        end_vertex_type: str,
        end_props: Optional[Dict[str, Any]],
    ) -> None:
        properties = {
            "start_id": start_id,
            "start_vertex_type": start_vertex_type,
            "end_id": end_id,
            "end_vertex_type": end_vertex_type,
            **get_timestamp(),
            "schema_version": "1",
        }
        super().__init__(id=id, properties=properties)


class UsedIn(Edge):
    def __init__(
        self,
        id: str,
        start_id: str,
        start_vertex_type: str,
        start_props: Optional[Dict[str, Any]],
        end_id: str,
        end_vertex_type: str,
        end_props: Optional[Dict[str, Any]],
    ) -> None:
        properties = {
            "start_id": start_id,
            "start_vertex_type": start_vertex_type,
            "end_id": end_id,
            "end_vertex_type": end_vertex_type,
            **get_timestamp(),
            "schema_version": "1",
        }
        super().__init__(id=id, properties=properties)


class Produce(Edge):
    def __init__(
        self,
        id: str,
        start_id: str,
        start_vertex_type: str,
        start_props: Optional[Dict[str, Any]],
        end_id: str,
        end_vertex_type: str,
        end_props: Optional[Dict[str, Any]],
    ) -> None:
        properties = {
            "start_id": start_id,
            "start_vertex_type": start_vertex_type,
            **(start_props or {}),
            "end_id": end_id,
            "end_vertex_type": end_vertex_type,
            **(end_props or {}),
        }
        super().__init__(id=id, properties=properties)


class KNOWS(Edge):
    def __init__(
        self,
        id: str,
        start_id: str,
        start_vertex_type: str,
        start_props: Optional[Dict[str, Any]],
        end_id: str,
        end_vertex_type: str,
        end_props: Optional[Dict[str, Any]],
    ) -> None:
        properties = {
            "start_id": start_id,
            "start_vertex_type": start_vertex_type,
            **(start_props or {}),
            "end_id": end_id,
            "end_vertex_type": end_vertex_type,
            **(end_props or {}),
        }
        super().__init__(id=id, properties=properties)


class LIKES(Edge):
    def __init__(
        self,
        id: str,
        start_id: str,
        start_vertex_type: str,
        start_props: Optional[Dict[str, Any]],
        end_id: str,
        end_vertex_type: str,
        end_props: Optional[Dict[str, Any]],
    ) -> None:
        properties = {
            "start_id": start_id,
            "start_vertex_type": start_vertex_type,
            **(start_props or {}),
            "end_id": end_id,
            "end_vertex_type": end_vertex_type,
            **(end_props or {}),
        }
        super().__init__(id=id, properties=properties)


def generate_base58_dummy_data(length: int) -> str:
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
    data_dir: str,
    file_name: str,
    data: List[Node],
) -> str:
    """
    Write a list of Node/Edge objects to a CSV file.

    Args:
        data_dir (str): Directory where the CSV file will be saved.
        file_name (str): Name of the CSV file.
        data (List[Node]): List of objects to write.

    Returns:
        str: The absolute path to the created CSV file.
    """
    if data is None or len(data) == 0:
        return ""

    # Build absolute path using data_dir
    base_dir = os.path.abspath(data_dir)
    os.makedirs(base_dir, exist_ok=True)
    file_path = os.path.join(base_dir, f"{file_name.lower()}.csv")

    headers = ['"id"'] + [f'"{k}"' for k in (data[0].properties or {}).keys()]
    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(",".join(headers) + "\n")
        lines = [
            ",".join([f'"{d.id}"'] + [f'"{v}"' for v in (d.properties or {}).values()])
            for d in data
        ]
        await f.write("\n".join(lines))

    return file_path


# Registry of classes for dynamic lookup
CLASSES: Dict[str, Type[Any]] = {
    "AirPort": AirPort,
    "Customer": Customer,
    "Product": Product,
    "Person": Person,
    "Manufacturer": Manufacturer,
    "Country": Country,
    "City": City,
    "BitcoinAddress": BitcoinAddress,
    "Cookie": Cookie,
    "IP": IP,
    "Phone": Phone,
    "Email": Email,
    "Payment": Payment,
    "PartnerEndUser": PartnerEndUser,
    "CreditCard": CreditCard,
    "CryptoAddress": CryptoAddress,
    "AirRoute": AirRoute,
    "Bought": Bought,
    "Has": Has,
    "PerformedBy": PerformedBy,
    "UsedBy": UsedBy,
    "UsedIn": UsedIn,
    "Produce": Produce,
    "KNOWS": KNOWS,
    "LIKES": LIKES,
}


async def generate_complete_data(
    edge_cls_name: str,
    edge_props: Dict[str, Any],
    nodes: Dict[str, int],
    data_dir: str,
) -> None:
    """
    Generate a set of edge objects (single source) and write them to CSV.

    Args:
        edge_cls_name (str): Name of the edge class.
        edge_props (Dict[str, Any]): Dictionary containing count, start, and end node type.
        nodes (Dict[str, int]): Dictionary with node types and their counts.
        data_dir (str): Directory to save CSV.

    Returns:
        None
    """
    print(f"Generating {edge_cls_name}: {edge_props['count']}...")

    start_node_cls = CLASSES[edge_props["start"]]
    end_node_cls = CLASSES[edge_props["end"]]
    start_node_data = [
        start_node_cls(id=str(i + 1)) for i in range(nodes[edge_props["start"]])
    ]
    end_node_data = [
        end_node_cls(id=str(i + 1)) for i in range(nodes[edge_props["end"]])
    ]

    data = []
    for i in range(edge_props["count"]):
        start = random.choice(start_node_data)
        end = random.choice(end_node_data)
        edge_instance = CLASSES[edge_cls_name](
            id=str(i + 1),
            start_id=start.id,
            start_vertex_type=edge_props["start"],
            start_props=start.properties,
            end_id=end.id,
            end_vertex_type=edge_props["end"],
            end_props=end.properties,
        )
        data.append(edge_instance)
    await put_csv(
        data_dir=data_dir,
        file_name=f"{edge_props['start']}_{edge_props['end']}_{edge_cls_name}",
        data=data,
    )


async def generate_nodes(
    cls_name: str, count: int, data_dir: str
) -> Tuple[str, List[Any]]:
    """
    Generate node objects and write them to CSV.

    Args:
        cls_name (str): Name of the node class.
        count (int): Number of objects to generate.
        data_dir (str): Directory to save CSV.

    Returns:
        Tuple: Class name and list of generated objects.
    """
    print(f"Generating {cls_name}: {count}...")
    node_cls = CLASSES[cls_name]
    data = [node_cls(id=str(i + 1)) for i in range(count)]
    await put_csv(data_dir=data_dir, file_name=cls_name, data=data)
    return cls_name, data


async def generate_edges(
    cls_name: str,
    prop_list: List[Dict[str, Any]],
    nodes_data: Dict[str, List[Any]],
    data_dir: str,
) -> None:
    """
    Generate edge objects (multiple source) and write them to CSV.

    Args:
        cls_name (str): Name of the edge class.
        prop_list (List[Dict[str, Any]]): List of edge property dictionaries.
        nodes_data (Dict[str, List[Any]]): Generated node data keyed by node type.
        data_dir (str): Directory to save CSV.

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
            CLASSES[cls_name](
                id=str(i + 1),
                start_id=random.choice(nodes_data[props["start"]]).id,
                start_vertex_type=start_vertex_type,
                start_props=random.choice(nodes_data[props["start"]]).properties,
                end_id=random.choice(nodes_data[props["end"]]).id,
                end_vertex_type=end_vertex_type,
                end_props=random.choice(nodes_data[props["end"]]).properties,
            )
            for i in range(props["count"])
        ]
        await put_csv(
            data_dir=data_dir,
            file_name=f"{cls_name}_{props['start']}_{props['end']}",
            data=data,
        )


async def main(pattern_no: int = 1, log_level: int = logging.INFO) -> None:
    """
    Main function to generate dummy data for a given pattern number.

    Args:
        pattern_no (int): The pattern number to generate data for.
        log_level (int): The logging level to use.
    """
    log.setLevel(log_level)

    # Configuration constants
    SINGLE_SOURCE = 1
    MULTI_SOURCE = 2

    data_dir = "generated_dummy_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    if not os.path.exists(data_dir):
        print(f"Creating directory {data_dir}")
        os.makedirs(data_dir)

    edges: Dict[str, List[Dict[str, Any]]] = {}
    match pattern_no:
        case 1:
            # Transaction, small data, multiple type of nodes and single type of edges
            source_type = SINGLE_SOURCE
            sub_dir = "/transaction"
            nodes = {"Customer": 10000, "Product": 1000}
            edges = {
                "Bought": [{"count": 20050, "start": "Customer", "end": "Product"}]
            }
        case 2:
            # Persons, small data, single type of nodes and multiple types of edges
            source_type = MULTI_SOURCE
            sub_dir = "/persons"
            nodes = {"Person": 1000}
            edges = {
                "KNOWS": [{"count": 1000, "start": "Person", "end": "Person"}],
                "LIKES": [{"count": 1000, "start": "Person", "end": "Person"}],
            }
        case 3:
            # Airroute, small data, single type of nodes and edges
            source_type = MULTI_SOURCE
            sub_dir = "/airroute"
            nodes = {"AirPort": 3500}
            edges = {
                "AirRoute": [{"count": 20000, "start": "AirPort", "end": "AirPort"}]
            }
        case 4:
            # Countries, small data, multiple types of nodes and single type of edges
            source_type = MULTI_SOURCE
            sub_dir = "/countries"
            nodes = {"Country": 200, "City": 10000}
            edges = {"Has": [{"count": 10000, "start": "Country", "end": "City"}]}
        case 5 | 6:
            # Payment, large data, multiple types of nodes and edges
            source_type = MULTI_SOURCE
            if pattern_no == 5:
                DIVIDER = 10000
                sub_dir = "/payment_small"
            else:
                DIVIDER = 10
                sub_dir = f"{data_dir}/payment_large"

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
                    {
                        "count": 60000000 // DIVIDER,
                        "start": "CreditCard",
                        "end": "Payment",
                    },
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
                    {
                        "count": 10000000 // DIVIDER,
                        "start": "CreditCard",
                        "end": "Payment",
                    },
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
                    {
                        "count": 80000000 // DIVIDER,
                        "start": "CreditCard",
                        "end": "Payment",
                    },
                ],
            }
        case _:
            raise ValueError("Invalid TEST_PATTERN specified.")

    if not os.path.exists(f"{data_dir}{sub_dir}"):
        print(f"Creating directory {data_dir}{sub_dir}")
        os.makedirs(f"{data_dir}{sub_dir}", exist_ok=True)

    if source_type == SINGLE_SOURCE:
        # Process single source edges
        for edge_cls_name, edge_props in edges.items():
            await generate_complete_data(
                edge_cls_name=edge_cls_name,
                edge_props=edge_props[0],
                nodes=nodes,
                data_dir=f"{data_dir}{sub_dir}",
            )
    elif source_type == MULTI_SOURCE:
        # Process multiple source nodes and edges
        node_tasks = [
            generate_nodes(cls_name, count, f"{data_dir}{sub_dir}")
            for cls_name, count in nodes.items()
        ]
        nodes_list = await asyncio.gather(*node_tasks)
        nodes_data = {cls_name: data for cls_name, data in nodes_list}

        edge_tasks = [
            generate_edges(cls_name, prop_list, nodes_data, f"{data_dir}{sub_dir}")
            for cls_name, prop_list in edges.items()
        ]
        await asyncio.gather(*edge_tasks)
