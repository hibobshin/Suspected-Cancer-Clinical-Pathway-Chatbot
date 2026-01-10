"""
ArangoDB database connection and utilities.

Provides connection pooling, collection management, and query helpers.
All database operations are logged for observability.
"""

from contextlib import contextmanager
from typing import Any, Generator

from arango import ArangoClient
from arango.database import StandardDatabase
from arango.exceptions import (
    ArangoServerError,
    CollectionCreateError,
    DatabaseCreateError,
)

from config.config import get_settings
from config.logging_config import get_logger

logger = get_logger(__name__)

# Singleton client instance
_client: ArangoClient | None = None
_db: StandardDatabase | None = None


def get_client() -> ArangoClient:
    """
    Get or create the ArangoDB client singleton.
    
    Returns:
        ArangoClient instance.
    """
    global _client
    if _client is None:
        settings = get_settings()
        _client = ArangoClient(hosts=settings.arango_host)
        logger.info("ArangoDB client initialized", host=settings.arango_host)
    return _client


def get_database() -> StandardDatabase:
    """
    Get or create the database connection.
    
    Creates the database if it doesn't exist.
    
    Returns:
        StandardDatabase instance.
    """
    global _db
    if _db is None:
        settings = get_settings()
        client = get_client()
        
        # Connect to system database to create our database if needed
        sys_db = client.db(
            "_system",
            username=settings.arango_username,
            password=settings.arango_password,
        )
        
        # Create database if it doesn't exist
        if not sys_db.has_database(settings.arango_database):
            try:
                sys_db.create_database(settings.arango_database)
                logger.info("Created database", database=settings.arango_database)
            except DatabaseCreateError as e:
                logger.error("Failed to create database", error=str(e))
                raise
        
        # Connect to our database
        _db = client.db(
            settings.arango_database,
            username=settings.arango_username,
            password=settings.arango_password,
        )
        logger.info("Connected to database", database=settings.arango_database)
        
        # Initialize collections
        _init_collections(_db)
    
    return _db


def _init_collections(db: StandardDatabase) -> None:
    """
    Initialize required collections if they don't exist.
    
    Args:
        db: The database instance.
    """
    collections = [
        # Conversations storage
        {"name": "conversations", "schema": None},
        # Chat messages
        {"name": "messages", "schema": None},
        # Pathway routes/modes
        {"name": "pathway_routes", "schema": None},
        # User sessions (optional)
        {"name": "sessions", "schema": None},
        # Audit log
        {"name": "audit_log", "schema": None},
    ]
    
    for col_config in collections:
        name = col_config["name"]
        if not db.has_collection(name):
            try:
                db.create_collection(name)
                logger.info("Created collection", collection=name)
            except CollectionCreateError as e:
                logger.warning("Collection creation failed", collection=name, error=str(e))


def close_connection() -> None:
    """Close the database connection."""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None
        logger.info("Database connection closed")


@contextmanager
def get_db_session() -> Generator[StandardDatabase, None, None]:
    """
    Context manager for database sessions.
    
    Yields:
        StandardDatabase instance.
    """
    db = get_database()
    try:
        yield db
    except ArangoServerError as e:
        logger.error("Database error", error=str(e))
        raise


# ============================================================================
# Collection helpers
# ============================================================================

def insert_document(collection: str, document: dict[str, Any]) -> dict[str, Any]:
    """
    Insert a document into a collection.
    
    Args:
        collection: Collection name.
        document: Document to insert.
        
    Returns:
        Inserted document with _key and _id.
    """
    db = get_database()
    result = db.collection(collection).insert(document, return_new=True)
    logger.debug("Document inserted", collection=collection, key=result["_key"])
    return result["new"]


def get_document(collection: str, key: str) -> dict[str, Any] | None:
    """
    Get a document by key.
    
    Args:
        collection: Collection name.
        key: Document key.
        
    Returns:
        Document or None if not found.
    """
    db = get_database()
    try:
        return db.collection(collection).get(key)
    except Exception:
        return None


def update_document(
    collection: str, key: str, updates: dict[str, Any]
) -> dict[str, Any] | None:
    """
    Update a document.
    
    Args:
        collection: Collection name.
        key: Document key.
        updates: Fields to update.
        
    Returns:
        Updated document or None if not found.
    """
    db = get_database()
    try:
        result = db.collection(collection).update(
            {"_key": key, **updates}, return_new=True
        )
        return result["new"]
    except Exception as e:
        logger.warning("Update failed", collection=collection, key=key, error=str(e))
        return None


def query_documents(
    aql: str, bind_vars: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """
    Execute an AQL query.
    
    Args:
        aql: AQL query string.
        bind_vars: Query bind variables.
        
    Returns:
        List of documents.
    """
    db = get_database()
    cursor = db.aql.execute(aql, bind_vars=bind_vars or {})
    return list(cursor)
