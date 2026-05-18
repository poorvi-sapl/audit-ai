"""
Phase 0 smoke test — verifies all four services are reachable.
Run after: make up && make smoke
"""
import os

import psycopg2
import pymongo
import pytest
import redis
from qdrant_client import QdrantClient


def _pg_dsn() -> dict:
    return dict(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "auditai"),
        user=os.getenv("POSTGRES_USER", "auditai"),
        password=os.getenv("POSTGRES_PASSWORD", "auditai_dev"),
        connect_timeout=5,
    )


def test_postgres_reachable():
    conn = psycopg2.connect(**_pg_dsn())
    cur = conn.cursor()
    cur.execute("SELECT 1")
    assert cur.fetchone()[0] == 1
    conn.close()


def test_qdrant_reachable():
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        timeout=5,
    )
    result = client.get_collections()
    assert result is not None


def test_redis_reachable():
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD") or None,
        socket_connect_timeout=5,
    )
    assert r.ping() is True


def test_mongodb_reachable():
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
    result = client.admin.command("ping")
    assert result.get("ok") == 1.0
    client.close()
