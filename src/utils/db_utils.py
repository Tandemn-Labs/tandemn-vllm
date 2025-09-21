from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import motor.motor_asyncio

from src.config.settings import MONGODB_DB_NAME, MONGODB_URI, PEER_CLEANUP_THRESHOLD

# MongoDB connection
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# Collection names
PEERS_COLLECTION = "peers"
METRICS_COLLECTION = "metrics"
BATCH_PROCESSING_COLLECTION = "batch_processing"


async def setup_collections():
    """Initialize MongoDB collections with proper indexes and time-series configuration."""
    # Create time-series collection for metrics if it doesn't exist
    collections = await db.list_collection_names()
    if "metrics" not in collections:
        await db.create_collection(
            "metrics",
            timeseries={
                "timeField": "timestamp",
                "metaField": "peer_id",
                "granularity": "seconds",
            },
        )
        # Create index on peer_id and timestamp
        await db[METRICS_COLLECTION].create_index([("peer_id", 1), ("timestamp", -1)])
        # Create TTL index with partialFilterExpression
        await db[METRICS_COLLECTION].create_index(
            "timestamp",
            expireAfterSeconds=86400,  # 24 hours
            partialFilterExpression={"peer_id": {"$exists": True}},
        )

    # Create indexes for peers collection
    await db[PEERS_COLLECTION].create_index("peer_id", unique=True)
    await db[PEERS_COLLECTION].create_index("last_seen")
    # create index for batch processing collection
    await db[BATCH_PROCESSING_COLLECTION].create_index(
        "file_id", unique=True, expireAfterSeconds=86400
    )


def has_significant_change(
    current_metrics: Dict[str, Any], last_metrics: Dict[str, Any]
) -> bool:
    """
    Check if metrics have changed significantly (>5% change).

    Args:
        current_metrics: Current metrics
        last_metrics: Last stored metrics

    Returns:
        True if significant change detected, False otherwise
    """
    if not last_metrics:
        return True

    # Check CPU and RAM changes
    if abs(current_metrics["cpu"] - last_metrics["cpu"]) > 5:
        return True
    if abs(current_metrics["ram"] - last_metrics["ram"]) > 5:
        return True

    # Check GPU changes
    current_gpus = current_metrics.get("gpu_info", [])
    last_gpus = last_metrics.get("gpu_info", [])

    if len(current_gpus) != len(last_gpus):
        return True

    for curr_gpu, last_gpu in zip(current_gpus, last_gpus):
        # Check VRAM usage change
        if (
            abs(curr_gpu["used_vram_gb"] - last_gpu["used_vram_gb"]) > 0.5
        ):  # 0.5 GB threshold
            return True
        # Check utilization change
        if (
            curr_gpu["utilization_percent"] is not None
            and last_gpu["utilization_percent"] is not None
            and abs(curr_gpu["utilization_percent"] - last_gpu["utilization_percent"])
            > 5
        ):
            return True
        # Check temperature change
        if (
            curr_gpu["temperature_celsius"] is not None
            and last_gpu["temperature_celsius"] is not None
            and abs(curr_gpu["temperature_celsius"] - last_gpu["temperature_celsius"])
            > 2
        ):
            return True

    return False


async def register_peer(peer_id: str, hostname: str):
    """
    Register a new peer in the pipeline.

    Args:
        peer_id: The unique identifier of the peer
        hostname: The hostname of the peer machine
    """
    await db[PEERS_COLLECTION].update_one(
        {"peer_id": peer_id},
        {
            "$set": {
                "peer_id": peer_id,
                "hostname": hostname,
                "last_seen": datetime.utcnow(),
                "is_active": True,
                "last_metrics": None,  # Store last metrics for change detection
            }
        },
        upsert=True,
    )


async def deregister_peer(peer_id: str):
    """
    Mark a peer as inactive in the pipeline.

    Args:
        peer_id: The unique identifier of the peer
    """
    await db[PEERS_COLLECTION].update_one(
        {"peer_id": peer_id},
        {"$set": {"is_active": False, "last_seen": datetime.utcnow()}},
    )


async def get_active_peers() -> List[str]:
    """
    Get all active peers in the pipeline.

    Returns:
        List of active peer IDs in order of registration
    """
    cursor = db[PEERS_COLLECTION].find(
        {"is_active": True},
        sort=[("_id", 1)],  # Sort by insertion order
    )
    peers = await cursor.to_list(length=None)
    return [peer["peer_id"] for peer in peers]


async def update_peer_metrics(peer_id: str, metrics: Dict[str, Any]):
    """
    Update metrics for a peer using time-series collection.
    Only updates if there's a significant change in metrics.

    Args:
        peer_id: The unique identifier of the peer
        metrics: Dictionary containing peer metrics
    """
    # Get last metrics for change detection
    peer = await db[PEERS_COLLECTION].find_one({"peer_id": peer_id})
    _ = peer.get("last_metrics") if peer else None

    # Check if metrics have changed significantly
    # if has_significant_change(metrics, last_metrics):
    # Insert new metrics document
    await db[METRICS_COLLECTION].insert_one(
        {"peer_id": peer_id, "timestamp": datetime.utcnow(), "metrics": metrics}
    )

    # Update last metrics in peers collection
    await db[PEERS_COLLECTION].update_one(
        {"peer_id": peer_id}, {"$set": {"last_metrics": metrics}}
    )


async def get_peer_metrics(
    peer_id: str, time_window: int = 300
) -> List[Dict[str, Any]]:
    """
    Get metrics for a peer over a time window.

    Args:
        peer_id: The unique identifier of the peer
        time_window: Time window in seconds (default: 5 minutes)

    Returns:
        List of metrics documents
    """
    cursor = db[METRICS_COLLECTION].find(
        {
            "peer_id": peer_id,
            "timestamp": {"$gte": datetime.utcnow() - timedelta(seconds=time_window)},
        },
        sort=[("timestamp", -1)],
    )
    return await cursor.to_list(length=None)


async def cleanup_inactive_peers(hours: int = PEER_CLEANUP_THRESHOLD):
    """
    Remove peers that have been inactive for too long.

    Args:
        hours: Maximum age in hours before removing inactive peers
    """
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    await db[PEERS_COLLECTION].delete_many(
        {"is_active": False, "last_seen": {"$lt": cutoff_time}}
    )


async def get_peer_status(peer_id: str) -> Optional[Dict[str, Any]]:
    """
    Get current status of a peer.

    Args:
        peer_id: The peer ID to get status for

    Returns:
        Dictionary containing peer status or None if peer not found
    """
    peer = await db[PEERS_COLLECTION].find_one({"peer_id": peer_id})
    if not peer:
        return None

    # Get latest metrics
    metrics = await get_peer_metrics(peer_id, time_window=60)
    latest_metrics = metrics[0]["metrics"] if metrics else {}

    return {
        "peer_id": peer_id,
        "hostname": peer.get("hostname"),
        "is_active": peer.get("is_active", False),
        "last_seen": peer["last_seen"],
        "metrics": latest_metrics,
    }


async def save_csv_processing_state_by_file_id(
    file_id: str,
    task_id: str,
    new_state: Dict[str, Any],
):
    """
    Save the processing state of a CSV file by file_id.
    """
    await db[BATCH_PROCESSING_COLLECTION].update_one(
        {"task_id": task_id},
        {
            "$set": {
                "file_id": file_id,
                "task_id": task_id,
                "next_byte_position": new_state["next_byte_position"],
                "last_processed_line": new_state["last_processed_line"],
                "batch_count": new_state["batch_count"],
                "timestamp": datetime.utcnow(),
            }
        },
        upsert=True,
    )


async def get_csv_processing_state_by_file_id(
    file_id: str, task_id: str
) -> Optional[Dict[str, Any]]:
    """
    Get the processing state of a CSV file by file_id.
    """
    # find the current state of the file
    current_state = await db[BATCH_PROCESSING_COLLECTION].find_one(
        {"file_id": file_id, "task_id": task_id}
    )
    # make the data structured if found
    if current_state:
        return {
            "file_id": current_state["file_id"],
            "task_id": current_state["task_id"],
            "next_byte_position": current_state["next_byte_position"],
            "last_processed_line": current_state["last_processed_line"],
            "batch_count": current_state["batch_count"],
        }
    # else don't return anything
    return None
