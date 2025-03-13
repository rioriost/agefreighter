#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from faker import Faker

# Configure logging; default to INFO
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

fake = Faker()

# Dedicated thread pool for CSV writing
csv_executor = ThreadPoolExecutor(max_workers=4)

BULK_SUPPORTED_NODE_TYPES = {"Product"}


def get_timestamp() -> Dict[str, Any]:
    dt = fake.date_time_this_year(before_now=True, after_now=False, tzinfo=None)
    return {"available_since": dt, "inserted_at": dt}


def prefixed_props(prefix: str, props: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if props is None:
        return {}
    return {f"{prefix}{k}": v for k, v in props.items()}


def generate_base58_dummy_data(length: int) -> str:
    base58_chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    return "".join(random.choice(base58_chars) for _ in range(length))


# ------------------ Lightweight Node Generators ------------------
# Standard generators return a dictionary representing a node.
def generate_airport(id: str) -> dict:
    return {
        "id": id,
        "Name": f"{fake.city()} Airport",
        "City": fake.city(),
        "Country": fake.country(),
        "IATA": fake.unique.bothify(text="???").upper(),
        "ICAO": fake.unique.bothify(text="????").upper(),
        "Latitude": fake.latitude(),
        "Longitude": fake.longitude(),
        "Altitude": fake.random_int(min=0, max=1000),
        "Timezone": fake.timezone(),
        "DST": random.choice(["E", "A", "S", "O", "Z", "N", "U"]),
        "Tz": fake.timezone(),
    }


def generate_customer(id: str) -> dict:
    return {
        "id": id,
        "Name": fake.name(),
        "Address": fake.address().replace("\n", " "),
        "Email": fake.email(),
        "Phone": fake.phone_number(),
    }


# Bulk generator for Product nodes using vectorized numeric operations.
def generate_product_bulk(ids: List[str]) -> List[dict]:
    n = len(ids)
    # Vectorized numeric generation:
    # Generate 5-digit numbers then divide by 100 for price in range ~[100, 1000)
    prices = np.random.randint(10000, 100000, size=n) / 100.0
    weights = np.random.randint(100, 1001, size=n)
    sizes = np.random.choice(["S", "M", "L", "XL"], size=n)
    result = []
    for i, id in enumerate(ids):
        result.append(
            {
                "id": id,
                "Phrase": fake.catch_phrase(),
                "SKU": fake.ean13(),
                "Price": round(prices[i], 2),
                "Color": fake.color_name(),
                "Size": sizes[i],
                "Weight": int(weights[i]),
            }
        )
    return result


def generate_person(id: str) -> dict:
    return {"id": id, "Name": fake.name()}


def generate_manufacturer(id: str) -> dict:
    return {
        "id": id,
        "Name": fake.company(),
        "Address": fake.address(),
        "Phone": fake.phone_number(),
        "Email": fake.email(),
        "Website": fake.url(),
    }


def generate_country(id: str) -> dict:
    return {
        "id": id,
        "Name": fake.country(),
        "Capital": fake.city(),
        "Population": fake.random_int(min=1_000_000, max=1_000_000_000),
        "ISO": fake.unique.bothify(text="??").upper(),
        "TLD": fake.unique.bothify(text="???").lower(),
        "FlagURL": fake.image_url(),
    }


def generate_city(id: str) -> dict:
    return {
        "id": id,
        "Name": fake.city(),
        "Latitude": fake.latitude(),
        "Longitude": fake.longitude(),
    }


def generate_bitcoinaddress(id: str) -> dict:
    base_props = get_timestamp()
    extra = {
        "address": random.choice(["cryptoaddress:", "bitcoinaddress:"])
        + generate_base58_dummy_data(16),
        "schema_version": "1",
    }
    base_props.update(extra)
    base_props["id"] = id
    return base_props


def generate_cookie(id: str) -> dict:
    base_props = get_timestamp()
    base_props.update(
        {
            "uaid": fake.uuid4(),
            "schema_version": "1",
        }
    )
    base_props["id"] = id
    return base_props


def generate_ip(id: str) -> dict:
    base_props = get_timestamp()
    base_props.update(
        {
            "address": fake.ipv4(),
            "schema_version": "1",
        }
    )
    base_props["id"] = id
    return base_props


def generate_phone(id: str) -> dict:
    base_props = get_timestamp()
    base_props.update(
        {
            "address": fake.ipv4(),
            "schema_version": "1",
        }
    )
    base_props["id"] = id
    return base_props


def generate_email(id: str) -> dict:
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
    base_props["id"] = id
    return base_props


def generate_payment(id: str) -> dict:
    base_props = get_timestamp()
    base_props.update(
        {
            "payment_id": fake.uuid4(),
            "schema_version": "1",
        }
    )
    base_props["id"] = id
    return base_props


def generate_partnerenduser(id: str) -> dict:
    base_props = get_timestamp()
    base_props.update(
        {
            "partner_end_user_id": fake.uuid4(),
            "schema_version": "1",
        }
    )
    base_props["id"] = id
    return base_props


def generate_creditcard(id: str) -> dict:
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
    base_props["id"] = id
    return base_props


def generate_cryptoaddress(id: str) -> dict:
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
    base_props["id"] = id
    return base_props


NODE_GENERATORS: Dict[str, Any] = {
    "AirPort": generate_airport,
    "Customer": generate_customer,
    # Use the bulk generator for products.
    "Product": generate_product_bulk,
    "Person": generate_person,
    "Manufacturer": generate_manufacturer,
    "Country": generate_country,
    "City": generate_city,
    "BitcoinAddress": generate_bitcoinaddress,
    "Cookie": generate_cookie,
    "IP": generate_ip,
    "Phone": generate_phone,
    "Email": generate_email,
    "Payment": generate_payment,
    "PartnerEndUser": generate_partnerenduser,
    "CreditCard": generate_creditcard,
    "CryptoAddress": generate_cryptoaddress,
}


# ------------------ Lightweight Edge Generators ------------------
def generate_airroute(
    id: str,
    start_id: str,
    start_type: str,
    start_props: dict,
    end_id: str,
    end_type: str,
    end_props: dict,
) -> dict:
    # This function is kept for single-edge generation.
    return {
        "id": id,
        "start_id": start_id,
        "start_vertex_type": start_type,
        **prefixed_props("start_", start_props),
        "end_id": end_id,
        "end_vertex_type": end_type,
        **prefixed_props("end_", end_props),
        "distance": fake.random_int(min=100, max=16000),
    }


# Bulk generator for AirRoute edges using vectorized operations.
def generate_airroute_bulk(
    start_idx: int,
    end_idx: int,
    edge_props: Dict[str, Any],
    start_node_data: List[Dict[str, Any]],
    end_node_data: List[Dict[str, Any]],
) -> List[dict]:
    n = end_idx - start_idx
    # Precompute random indices for start and end nodes.
    start_indices = np.random.randint(0, len(start_node_data), size=n)
    end_indices = np.random.randint(0, len(end_node_data), size=n)
    # Vectorized generation of distances.
    distances = np.random.randint(100, 16001, size=n)
    edges = []
    for i in range(n):
        start = start_node_data[start_indices[i]]
        end = end_node_data[end_indices[i]]
        edge = {
            "id": str(start_idx + i + 1),
            "start_id": start["id"],
            "start_vertex_type": edge_props["start"],
            **prefixed_props("start_", start),
            "end_id": end["id"],
            "end_vertex_type": edge_props["end"],
            **prefixed_props("end_", end),
            "distance": int(distances[i]),
        }
        edges.append(edge)
    return edges


def generate_bought(
    id: str,
    start_id: str,
    start_type: str,
    start_props: dict,
    end_id: str,
    end_type: str,
    end_props: dict,
) -> dict:
    return {
        "id": id,
        "start_id": start_id,
        "start_vertex_type": start_type,
        **(start_props or {}),
        "end_id": end_id,
        "end_vertex_type": end_type,
        **(end_props or {}),
    }


def generate_has(
    id: str,
    start_id: str,
    start_type: str,
    start_props: dict,
    end_id: str,
    end_type: str,
    end_props: dict,
) -> dict:
    return {
        "id": id,
        "start_id": start_id,
        "start_vertex_type": start_type,
        **prefixed_props("start_", start_props),
        "end_id": end_id,
        "end_vertex_type": end_type,
        **prefixed_props("end_", end_props),
        "since": fake.date_time(),
    }


def generate_performedby(
    id: str,
    start_id: str,
    start_type: str,
    start_props: dict,
    end_id: str,
    end_type: str,
    end_props: dict,
) -> dict:
    return {
        "id": id,
        "start_id": start_id,
        "start_vertex_type": start_type,
        "end_id": end_id,
        "end_vertex_type": end_type,
        **get_timestamp(),
    }


def generate_usedby(
    id: str,
    start_id: str,
    start_type: str,
    start_props: dict,
    end_id: str,
    end_type: str,
    end_props: dict,
) -> dict:
    return {
        "id": id,
        "start_id": start_id,
        "start_vertex_type": start_type,
        "end_id": end_id,
        "end_vertex_type": end_type,
        **get_timestamp(),
        "schema_version": "1",
    }


def generate_usedin(
    id: str,
    start_id: str,
    start_type: str,
    start_props: dict,
    end_id: str,
    end_type: str,
    end_props: dict,
) -> dict:
    return {
        "id": id,
        "start_id": start_id,
        "start_vertex_type": start_type,
        "end_id": end_id,
        "end_vertex_type": end_type,
        **get_timestamp(),
        "schema_version": "1",
    }


def generate_produce(
    id: str,
    start_id: str,
    start_type: str,
    start_props: dict,
    end_id: str,
    end_type: str,
    end_props: dict,
) -> dict:
    return {
        "id": id,
        "start_id": start_id,
        "start_vertex_type": start_type,
        **(start_props or {}),
        "end_id": end_id,
        "end_vertex_type": end_type,
        **(end_props or {}),
    }


def generate_knows(
    id: str,
    start_id: str,
    start_type: str,
    start_props: dict,
    end_id: str,
    end_type: str,
    end_props: dict,
) -> dict:
    return {
        "id": id,
        "start_id": start_id,
        "start_vertex_type": start_type,
        **(start_props or {}),
        "end_id": end_id,
        "end_vertex_type": end_type,
        **(end_props or {}),
        "since": datetime.now().isoformat(),
    }


def generate_likes(
    id: str,
    start_id: str,
    start_type: str,
    start_props: dict,
    end_id: str,
    end_type: str,
    end_props: dict,
) -> dict:
    return {
        "id": id,
        "start_id": start_id,
        "start_vertex_type": start_type,
        **(start_props or {}),
        "end_id": end_id,
        "end_vertex_type": end_type,
        **(end_props or {}),
        "since": datetime.now().isoformat(),
    }


EDGE_GENERATORS: Dict[str, Any] = {
    "AirRoute": generate_airroute,  # For single generation fallback.
    "Bought": generate_bought,
    "Has": generate_has,
    "PerformedBy": generate_performedby,
    "UsedBy": generate_usedby,
    "UsedIn": generate_usedin,
    "Produce": generate_produce,
    "KNOWS": generate_knows,
    "LIKES": generate_likes,
}


# ------------------ CSV Writing ------------------
def sync_put_csv(data_dir: str, file_name: str, data: List[Dict[str, Any]]) -> str:
    if not data:
        return ""
    base_dir = os.path.abspath(data_dir)
    os.makedirs(base_dir, exist_ok=True)
    file_path = os.path.join(base_dir, f"{file_name.lower()}.csv")

    # Create CSV headers.
    headers = [f'"{k}"' for k in data[0].keys()]

    # Choose a buffer size (number of lines per flush)
    buffer_size = 100000

    # Open the file with an explicit buffering size (e.g., 1MB)
    with open(file_path, "w", encoding="utf-8", buffering=1_048_576) as f:
        # Write header
        f.write(",".join(headers) + "\n")

        # Use a list to accumulate lines before writing
        buffer_lines = []
        for i, row in enumerate(data, start=1):
            line = ",".join([f'"{v}"' for v in row.values()])
            buffer_lines.append(line)
            if i % buffer_size == 0:
                f.write("\n".join(buffer_lines) + "\n")
                buffer_lines = []
        # Write any remaining lines
        if buffer_lines:
            f.write("\n".join(buffer_lines))
    return file_path


async def put_csv(data_dir: str, file_name: str, data: List[Dict[str, Any]]) -> str:
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        csv_executor, sync_put_csv, data_dir, file_name, data
    )
    return result


# ------------------ Multiprocessing Helpers ------------------
def generate_node_chunk(
    node_type: str, node_gen: Any, start_idx: int, end_idx: int
) -> List[Dict[str, Any]]:
    if node_type in BULK_SUPPORTED_NODE_TYPES:
        ids = [str(i + 1) for i in range(start_idx, end_idx)]
        return node_gen(ids)
    else:
        return [node_gen(str(i + 1)) for i in range(start_idx, end_idx)]


def generate_edge_chunk(
    start_idx: int,
    end_idx: int,
    edge_name: str,
    edge_props: Dict[str, Any],
    start_node_data: List[Dict[str, Any]],
    end_node_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # For AirRoute edges, use the bulk vectorized generator.
    if edge_name == "AirRoute":
        return generate_airroute_bulk(
            start_idx, end_idx, edge_props, start_node_data, end_node_data
        )
    else:
        gen = EDGE_GENERATORS[edge_name]
        chunk = []
        for i in range(start_idx, end_idx):
            start = random.choice(start_node_data)
            end = random.choice(end_node_data)
            edge = gen(
                str(i + 1),
                start["id"],
                edge_props["start"],
                start,
                end["id"],
                edge_props["end"],
                end,
            )
            chunk.append(edge)
        return chunk


# ------------------ Data Generation Functions ------------------
async def generate_complete_data(
    edge_name: str,
    edge_props: Dict[str, Any],
    nodes: Dict[str, int],
    data_dir: str,
) -> None:
    print(f"Generating {edge_name}: {edge_props['count']}...")
    start_node_gen = NODE_GENERATORS[edge_props["start"]]
    end_node_gen = NODE_GENERATORS[edge_props["end"]]
    if edge_props["start"] in BULK_SUPPORTED_NODE_TYPES:
        start_node_data = start_node_gen(
            [str(i + 1) for i in range(nodes[edge_props["start"]])]
        )
    else:
        start_node_data = [
            start_node_gen(str(i + 1)) for i in range(nodes[edge_props["start"]])
        ]

    if edge_props["end"] in BULK_SUPPORTED_NODE_TYPES:
        end_node_data = end_node_gen(
            [str(i + 1) for i in range(nodes[edge_props["end"]])]
        )
    else:
        end_node_data = [
            end_node_gen(str(i + 1)) for i in range(nodes[edge_props["end"]])
        ]
    total_edges = edge_props["count"]
    chunk_size = 100000  # Adjust as needed.
    tasks = []
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for start_idx in range(0, total_edges, chunk_size):
            end_idx = min(start_idx + chunk_size, total_edges)
            tasks.append(
                loop.run_in_executor(
                    executor,
                    generate_edge_chunk,
                    start_idx,
                    end_idx,
                    edge_name,
                    edge_props,
                    start_node_data,
                    end_node_data,
                )
            )
        results = await asyncio.gather(*tasks)
    data = [edge for sublist in results for edge in sublist]
    log.info(f"Writing {len(data)} edges.")
    await put_csv(
        data_dir=data_dir,
        file_name=f"{edge_props['start']}_{edge_props['end']}_{edge_name}",
        data=data,
    )


async def generate_nodes(
    cls_name: str, count: int, data_dir: str
) -> Tuple[str, List[Dict[str, Any]]]:
    print(f"Generating {cls_name}: {count}...")
    node_gen = NODE_GENERATORS[cls_name]
    chunk_size = 100000  # Adjust based on workload.
    tasks = []
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for start_idx in range(0, count, chunk_size):
            end_idx = min(start_idx + chunk_size, count)
            tasks.append(
                loop.run_in_executor(
                    executor,
                    generate_node_chunk,
                    cls_name,  # Pass the node type.
                    node_gen,
                    start_idx,
                    end_idx,
                )
            )
        results = await asyncio.gather(*tasks)
    data = [node for sublist in results for node in sublist]
    await put_csv(data_dir=data_dir, file_name=cls_name, data=data)
    return cls_name, data


async def generate_edges(
    edge_name: str,
    prop_list: List[Dict[str, Any]],
    nodes_data: Dict[str, List[Dict[str, Any]]],
    data_dir: str,
) -> None:
    loop = asyncio.get_running_loop()
    for props in prop_list:
        print(
            f"Generating {edge_name} {props['start']} - {props['end']}: {props['count']}..."
        )
        total_edges = props["count"]
        chunk_size = 100000  # Adjust based on workload.
        tasks = []
        start_node_data = nodes_data[props["start"]]
        end_node_data = nodes_data[props["end"]]
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            for start_idx in range(0, total_edges, chunk_size):
                end_idx = min(start_idx + chunk_size, total_edges)
                tasks.append(
                    loop.run_in_executor(
                        executor,
                        generate_edge_chunk,
                        start_idx,
                        end_idx,
                        edge_name,
                        props,
                        start_node_data,
                        end_node_data,
                    )
                )
            results = await asyncio.gather(*tasks)
        data = [edge for sublist in results for edge in sublist]
        await put_csv(
            data_dir=data_dir,
            file_name=f"{edge_name}_{props['start']}_{props['end']}",
            data=data,
        )


# ------------------ Main Function ------------------
async def main(
    pattern_no: int = 1, multiplier: int = 1, log_level: int = logging.INFO
) -> None:
    log.setLevel(log_level)
    if multiplier >= 100:
        log.warning("Multiplier is too large. It will generate a lot of data.")

    # Configuration constants
    SINGLE_SOURCE = 1
    MULTI_SOURCE = 2

    data_dir = "generated_dummy_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(data_dir):
        print(f"Creating directory {data_dir}")
        os.makedirs(data_dir)

    edges: Dict[str, List[Dict[str, Any]]] = {}
    sub_dir = ""
    nodes: Dict[str, int] = {}

    if pattern_no == 1:
        # Transaction: multiple node types, single edge type.
        source_type = SINGLE_SOURCE
        sub_dir = "/transaction"
        nodes = {"Customer": 10000 * multiplier, "Product": 1000 * multiplier}
        edges = {
            "Bought": [
                {"count": 20050 * multiplier, "start": "Customer", "end": "Product"}
            ]
        }
    elif pattern_no == 2:
        # Persons: single node type, multiple edge types.
        source_type = MULTI_SOURCE
        sub_dir = "/persons"
        nodes = {"Person": 1000 * multiplier}
        edges = {
            "KNOWS": [{"count": 1000 * multiplier, "start": "Person", "end": "Person"}],
            "LIKES": [{"count": 1000 * multiplier, "start": "Person", "end": "Person"}],
        }
    elif pattern_no == 3:
        # Airroute: single node and edge type.
        source_type = MULTI_SOURCE
        sub_dir = "/airroute"
        nodes = {"AirPort": 3500 * multiplier}
        edges = {
            "AirRoute": [
                {"count": 20000 * multiplier, "start": "AirPort", "end": "AirPort"}
            ]
        }
    elif pattern_no == 4:
        # Countries: multiple node types, single edge type.
        source_type = MULTI_SOURCE
        sub_dir = "/countries"
        nodes = {"Country": 200 * multiplier, "City": 10000 * multiplier}
        edges = {
            "Has": [{"count": 10000 * multiplier, "start": "Country", "end": "City"}]
        }
    elif pattern_no == 5:
        # Payment: multiple node types, multiple edge types.
        source_type = MULTI_SOURCE
        sub_dir = "/payment"
        nodes = {
            "BitcoinAddress": 900 * multiplier,
            "Cookie": 2700 * multiplier,
            "IP": 2200 * multiplier,
            "Phone": 960 * multiplier,
            "Email": 960 * multiplier,
            "Payment": 7000 * multiplier,
            "CreditCard": 1200 * multiplier,
            "PartnerEndUser": 400 * multiplier,
            "CryptoAddress": 160 * multiplier,
        }
        edges = {
            "UsedIn": [
                {"count": 6000 * multiplier, "start": "Cookie", "end": "Payment"},
                {"count": 6000 * multiplier, "start": "Email", "end": "Payment"},
                {
                    "count": 6000 * multiplier,
                    "start": "CryptoAddress",
                    "end": "Payment",
                },
                {"count": 6000 * multiplier, "start": "Phone", "end": "Payment"},
                {"count": 6000 * multiplier, "start": "CreditCard", "end": "Payment"},
            ],
            "PerformedBy": [
                {"count": 1000 * multiplier, "start": "Cookie", "end": "Payment"},
                {"count": 1000 * multiplier, "start": "Email", "end": "Payment"},
                {
                    "count": 1000 * multiplier,
                    "start": "CryptoAddress",
                    "end": "Payment",
                },
                {"count": 1000 * multiplier, "start": "Phone", "end": "Payment"},
                {"count": 1000 * multiplier, "start": "CreditCard", "end": "Payment"},
            ],
            "UsedBy": [
                {"count": 8000 * multiplier, "start": "Cookie", "end": "Payment"},
                {"count": 8000 * multiplier, "start": "Email", "end": "Payment"},
                {
                    "count": 8000 * multiplier,
                    "start": "CryptoAddress",
                    "end": "Payment",
                },
                {"count": 8000 * multiplier, "start": "Phone", "end": "Payment"},
                {"count": 8000 * multiplier, "start": "CreditCard", "end": "Payment"},
            ],
        }
    else:
        raise ValueError("Invalid TEST_PATTERN specified.")

    full_dir = f"{data_dir}{sub_dir}"
    if not os.path.exists(full_dir):
        print(f"Creating directory {full_dir}")
        os.makedirs(full_dir, exist_ok=True)

    if source_type == SINGLE_SOURCE:
        for edge_name, edge_props in edges.items():
            await generate_complete_data(
                edge_name=edge_name,
                edge_props=edge_props[0],
                nodes=nodes,
                data_dir=full_dir,
            )
    elif source_type == MULTI_SOURCE:
        node_tasks = [
            generate_nodes(cls_name, count, full_dir)
            for cls_name, count in nodes.items()
        ]
        nodes_list = await asyncio.gather(*node_tasks)
        nodes_data = {cls_name: data for cls_name, data in nodes_list}
        edge_tasks = [
            generate_edges(edge_name, prop_list, nodes_data, full_dir)
            for edge_name, prop_list in edges.items()
        ]
        await asyncio.gather(*edge_tasks)


if __name__ == "__main__":
    asyncio.run(main())
