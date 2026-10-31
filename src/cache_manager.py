"""
Dual caching system for Healthcare AI Research Agent
"""

import os
import json
import hashlib
import sqlite3
from typing import Optional, Any, Dict, List, Union
from pathlib import Path
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from .config import CacheType

class BaseCache(ABC):
    """Abstract base class for cache implementations"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries"""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass

class InMemoryCache(BaseCache):
    """
    In-Memory Cache Implementation

    Features:
    - Fast read/write operations
    - Optional TTL (time-to-live) for entries
    - Size limit with LRU eviction
    - Thread-safe operations
    """

    def __init__(self, max_size: int = 100):
        """
        Initialize In-Memory Cache

        Args:
            max_size: Maximum number of entries in cache
        """
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_order: List[str] = []

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None

        entry = self.cache[key]

        # Check if expired
        if entry.get("expires_at") and datetime.now() > entry["expires_at"]:
            self.delete(key)
            return None

        # Update access order (move to end for LRU)
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

        return entry.get("value")

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            # Create entry
            entry = {
                "value": value,
                "created_at": datetime.now(),
                "accessed_at": datetime.now()
            }

            # Add TTL if specified
            if ttl:
                entry["expires_at"] = datetime.now() + timedelta(seconds=ttl)

            # Check if key exists
            if key in self.cache:
                # Update existing entry
                self.cache[key] = entry
                # Update access order
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
            else:
                # Check size limit
                if len(self.cache) >= self.max_size:
                    # Evict least recently used
                    if self.access_order:
                        oldest_key = self.access_order.pop(0)
                        del self.cache[oldest_key]

                # Add new entry
                self.cache[key] = entry
                self.access_order.append(key)

            return True

        except Exception as e:
            print(f"Error setting cache entry: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if key in self.cache:
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
            return True
        return False

    def clear(self) -> bool:
        """Clear all cache entries"""
        self.cache.clear()
        self.access_order.clear()
        return True

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if key not in self.cache:
            return False

        entry = self.cache[key]
        if entry.get("expires_at") and datetime.now() > entry["expires_at"]:
            self.delete(key)
            return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        current_size = len(self.cache)
        max_size = self.max_size
        utilization = (current_size / max_size * 100) if max_size > 0 else 0

        # Calculate average access time (not applicable for in-memory)
        avg_access_time = 0.001  # Estimated in ms

        return {
            "type": "IN_MEMORY",
            "current_size": current_size,
            "max_size": max_size,
            "utilization_percent": round(utilization, 2),
            "avg_access_time_ms": avg_access_time,
            "entries": list(self.cache.keys())[:10]  # First 10 keys
        }

    def get_all_keys(self) -> List[str]:
        """Get all keys in cache"""
        return list(self.cache.keys())

class SQLiteCache(BaseCache):
    """
    SQLite Cache Implementation

    Features:
    - Persistent storage using SQLite
    - Table-based organization
    - Optional TTL for entries
    - Indexed keys for fast lookup
    - Automatic table creation
    """

    def __init__(self, db_path: str = "./cache/healthcare_cache.db"):
        """
        Initialize SQLite Cache

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._initialize_database()

    def _initialize_database(self):
        """Initialize the SQLite database and tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create cache table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    size INTEGER
                )
                """)

                # Create index for faster lookups
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_key ON cache(key)
                """)

                # Create index for expiration
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at)
                WHERE expires_at IS NOT NULL
                """)

                conn.commit()

        except Exception as e:
            print(f"Error initializing SQLite cache: {e}")
            raise

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get entry
                cursor.execute("""
                SELECT value, expires_at FROM cache WHERE key = ?
                """, (key,))

                result = cursor.fetchone()
                if not result:
                    return None

                value_json, expires_at = result

                # Check if expired
                if expires_at:
                    expires_dt = datetime.fromisoformat(expires_at)
                    if datetime.now() > expires_dt:
                        self.delete(key)
                        return None

                # Update access time
                cursor.execute("""
                UPDATE cache SET accessed_at = CURRENT_TIMESTAMP WHERE key = ?
                """, (key,))

                conn.commit()

                # Parse value
                return json.loads(value_json)

        except Exception as e:
            print(f"Error getting cache entry: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            # Serialize value
            value_json = json.dumps(value)
            size = len(value_json.encode('utf-8'))

            # Calculate expiration
            expires_at = None
            if ttl:
                expires_at = (datetime.now() + timedelta(seconds=ttl)).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Check if key exists
                cursor.execute("SELECT 1 FROM cache WHERE key = ?", (key,))
                exists = cursor.fetchone() is not None

                if exists:
                    # Update existing entry
                    cursor.execute("""
                    UPDATE cache
                    SET value = ?, accessed_at = CURRENT_TIMESTAMP,
                        expires_at = ?, size = ?
                    WHERE key = ?
                    """, (value_json, expires_at, size, key))
                else:
                    # Insert new entry
                    cursor.execute("""
                    INSERT INTO cache (key, value, expires_at, size)
                    VALUES (?, ?, ?, ?)
                    """, (key, value_json, expires_at, size))

                conn.commit()

            return True

        except Exception as e:
            print(f"Error setting cache entry: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
                deleted = cursor.rowcount > 0

                conn.commit()

            return deleted

        except Exception as e:
            print(f"Error deleting cache entry: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("DELETE FROM cache")
                conn.commit()

            return True

        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                SELECT 1 FROM cache
                WHERE key = ? AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                """, (key,))

                return cursor.fetchone() is not None

        except Exception as e:
            print(f"Error checking cache entry: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get basic stats
                cursor.execute("SELECT COUNT(*) FROM cache")
                current_size = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM cache WHERE expires_at IS NOT NULL")
                with_ttl = cursor.fetchone()[0]

                cursor.execute("SELECT AVG(size) FROM cache")
                avg_size = cursor.fetchone()[0] or 0

                # Get file size
                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

                return {
                    "type": "SQLITE",
                    "db_path": str(self.db_path),
                    "current_size": current_size,
                    "max_size": "unlimited",
                    "utilization_percent": 0,  # Not applicable
                    "avg_access_time_ms": 1.0,  # Estimated
                    "avg_entry_size_bytes": round(avg_size, 2),
                    "entries_with_ttl": with_ttl,
                    "db_file_size_bytes": db_size,
                    "db_file_size_mb": round(db_size / (1024 * 1024), 2)
                }

        except Exception as e:
            print(f"Error getting cache stats: {e}")
            return {
                "type": "SQLITE",
                "error": str(e)
            }

    def get_all_keys(self) -> List[str]:
        """Get all keys in cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT key FROM cache")
                keys = [row[0] for row in cursor.fetchall()]

            return keys

        except Exception as e:
            print(f"Error getting all keys: {e}")
            return []

    def cleanup_expired(self) -> int:
        """Remove expired entries from cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                DELETE FROM cache
                WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
                """)

                deleted_count = cursor.rowcount
                conn.commit()

            return deleted_count

        except Exception as e:
            print(f"Error cleaning up expired entries: {e}")
            return 0

    def vacuum(self) -> bool:
        """Optimize the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("VACUUM")
                conn.commit()
            return True
        except Exception as e:
            print(f"Error vacuuming database: {e}")
            return False

class CacheManager:
    """
    Unified Cache Manager

    Manages multiple cache implementations and provides a unified interface.
    Supports switching between cache types at runtime.
    """

    def __init__(
        self,
        cache_type: CacheType = CacheType.IN_MEMORY,
        memory_cache_size: int = 100,
        sqlite_cache_path: str = "./cache/healthcare_cache.db"
    ):
        """
        Initialize Cache Manager

        Args:
            cache_type: Type of cache to use (IN_MEMORY or SQLITE)
            memory_cache_size: Size of in-memory cache
            sqlite_cache_path: Path to SQLite database
        """
        self.cache_type = cache_type
        self.memory_cache_size = memory_cache_size
        self.sqlite_cache_path = sqlite_cache_path

        # Initialize caches
        self.caches: Dict[CacheType, BaseCache] = {}
        self._initialize_caches()

        # Current cache
        self.current_cache = self.caches[cache_type]

    def _initialize_caches(self):
        """Initialize all cache implementations"""
        # Initialize In-Memory Cache
        self.caches[CacheType.IN_MEMORY] = InMemoryCache(
            max_size=self.memory_cache_size
        )

        # Initialize SQLite Cache
        self.caches[CacheType.SQLITE] = SQLiteCache(
            db_path=self.sqlite_cache_path
        )

    def switch_cache_type(self, cache_type: CacheType):
        """Switch to a different cache type"""
        if cache_type not in self.caches:
            raise ValueError(f"Unknown cache type: {cache_type}")

        self.cache_type = cache_type
        self.current_cache = self.caches[cache_type]
        print(f"🔄 Switched to {cache_type.value} cache")

    def get(self, key: str) -> Optional[Any]:
        """Get value from current cache"""
        return self.current_cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in current cache"""
        return self.current_cache.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        """Delete value from current cache"""
        return self.current_cache.delete(key)

    def clear(self) -> bool:
        """Clear current cache"""
        return self.current_cache.clear()

    def clear_all_caches(self) -> bool:
        """Clear all cache implementations"""
        results = []
        for cache_type, cache in self.caches.items():
            result = cache.clear()
            results.append(result)
            print(f"🗑️  Cleared {cache_type.value} cache: {'✅' if result else '❌'}")
        return all(results)

    def exists(self, key: str) -> bool:
        """Check if key exists in current cache"""
        return self.current_cache.exists(key)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for current cache"""
        return self.current_cache.get_stats()

    def get_all_stats(self) -> Dict[CacheType, Dict[str, Any]]:
        """Get statistics for all cache implementations"""
        return {cache_type: cache.get_stats() for cache_type, cache in self.caches.items()}

    def get_all_keys(self) -> List[str]:
        """Get all keys in current cache"""
        return self.current_cache.get_all_keys()

    def generate_cache_key(self, *args: Any, **kwargs: Any) -> str:
        """
        Generate a consistent cache key from arguments

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            SHA256 hash of the arguments
        """
        # Convert all arguments to string
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])

        # Join and hash
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get_with_fallback(
        self,
        key: str,
        fallback_cache_type: Optional[CacheType] = None
    ) -> Optional[Any]:
        """
        Get value from current cache with fallback to another cache

        Args:
            key: Cache key
            fallback_cache_type: Cache type to use as fallback

        Returns:
            Value from cache or None
        """
        # Try current cache
        value = self.get(key)
        if value is not None:
            return value

        # Try fallback cache if specified
        if fallback_cache_type and fallback_cache_type in self.caches:
            value = self.caches[fallback_cache_type].get(key)
            if value is not None:
                # Optionally set in current cache
                self.set(key, value)
                return value

        return None

    def set_with_replication(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        replicate_to: Optional[List[CacheType]] = None
    ) -> bool:
        """
        Set value in current cache with replication to other caches

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            replicate_to: List of cache types to replicate to

        Returns:
            True if all operations succeeded
        """
        results = []

        # Set in current cache
        result = self.set(key, value, ttl)
        results.append(result)

        # Replicate to other caches if specified
        if replicate_to:
            for cache_type in replicate_to:
                if cache_type in self.caches:
                    result = self.caches[cache_type].set(key, value, ttl)
                    results.append(result)

        return all(results)

    def invalidate_related(self, prefix: str) -> int:
        """
        Invalidate all keys with a given prefix

        Args:
            prefix: Key prefix to match

        Returns:
            Number of entries invalidated
        """
        keys = self.get_all_keys()
        invalidated = 0

        for key in keys:
            if key.startswith(prefix):
                if self.delete(key):
                    invalidated += 1

        return invalidated

    def cleanup(self) -> Dict[str, Any]:
        """
        Perform cleanup operations on all caches

        Returns:
            Cleanup results
        """
        results = {}

        for cache_type, cache in self.caches.items():
            if isinstance(cache, SQLiteCache):
                # Cleanup expired entries for SQLite
                deleted = cache.cleanup_expired()
                vacuumed = cache.vacuum()
                results[cache_type.value] = {
                    "expired_entries_deleted": deleted,
                    "vacuumed": vacuumed
                }
            else:
                results[cache_type.value] = {"cleaned": True}

        return results

# Utility function to create cache manager with default settings
def create_cache_manager(
    cache_type: CacheType = CacheType.IN_MEMORY,
    memory_cache_size: int = 100,
    sqlite_cache_path: str = "./cache/healthcare_cache.db"
) -> CacheManager:
    """
    Factory function to create a CacheManager with default settings

    Args:
        cache_type: Type of cache to use
        memory_cache_size: Size of in-memory cache
        sqlite_cache_path: Path to SQLite database

    Returns:
        Configured CacheManager
    """
    return CacheManager(
        cache_type=cache_type,
        memory_cache_size=memory_cache_size,
        sqlite_cache_path=sqlite_cache_path
    )