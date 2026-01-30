"""
Direct PostgreSQL Database Module

This module provides direct PostgreSQL connection management using psycopg2
for reliable, persistent database connections. It replaces the Supabase PostgREST
client for database operations while maintaining a similar interface.

Usage:
    from config.db import db

    # Select
    rows = db.table("clients").select("id, name").eq("status", "active").execute()

    # Insert
    result = db.table("clients").insert({"name": "Test", "email": "test@example.com"}).execute()

    # Update
    db.table("clients").update({"status": "active"}).eq("id", client_id).execute()

    # Delete
    db.table("clients").delete().eq("id", client_id).execute()

    # RPC (stored procedure)
    result = db.rpc("match_questions", {"query_embedding": [...], "match_count": 5}).execute()
"""

import json
import time
import traceback
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager
from datetime import datetime, date
from decimal import Decimal

import psycopg2
from psycopg2 import pool, sql, extras
from psycopg2.extensions import connection as PgConnection


class DatabaseConfig:
    """Database configuration singleton"""
    _instance = None
    _pool: pool.ThreadedConnectionPool | None = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def initialize(self, database_url: str = None, **kwargs):
        """Initialize the database connection pool"""
        if self._initialized and self._pool is not None:
            return
        
        import os
        from urllib.parse import urlparse
        
        # Get connection parameters
        if database_url:
            parsed = urlparse(database_url)
            self.user = parsed.username or "postgres"
            self.password = parsed.password or ""
            self.host = parsed.hostname or "localhost"
            self.port = str(parsed.port or 5432)
            self.dbname = parsed.path.lstrip('/') if parsed.path else "postgres"
        else:
            # Support both individual env vars and kwargs
            self.user = kwargs.get('user') or os.getenv("DB_USER", "postgres")
            self.password = kwargs.get('password') or os.getenv("DB_PASSWORD", "")
            self.host = kwargs.get('host') or os.getenv("DB_HOST", "")
            self.port = str(kwargs.get('port') or os.getenv("DB_PORT", "5432"))
            self.dbname = kwargs.get('dbname') or os.getenv("DB_NAME", "postgres")
        
        if not self.host or not self.password:
            raise ValueError(
                "Database not configured. Set DATABASE_URL or individual DB_* environment variables."
            )
        # Detect if this is a Supabase connection (requires SSL)
        # Check host, dbname, dedicated IPv4, or explicit flag
        is_supabase = (
            "supabase" in self.host.lower() or 
            self.dbname == "postgres" or
            os.getenv("IS_SUPABASE", "1") == "1" or  # Default to true for this project
            os.getenv("USE_DEDICATED_IP", "0") == "1"  # Dedicated IPv4 also requires SSL
        )
        
        try:
            connection_params = {
                "minconn": 5,
                "maxconn": 40,
                "user": self.user,
                "password": self.password,
                "host": self.host,
                "port": self.port,
                "dbname": self.dbname,
                "connect_timeout": 10,
                "keepalives": 1,
                "keepalives_idle": 60,
                "keepalives_interval": 10,
                "keepalives_count": 5,
                "options": '-c statement_timeout=60000',  # 1 minute timeout
            }
            
            # Add SSL for Supabase connections (required)
            if is_supabase:
                connection_params["sslmode"] = "require"
            
            self._pool = pool.ThreadedConnectionPool(**connection_params)
            self._initialized = True
            ssl_info = " with SSL" if is_supabase else ""
            print(f"✓ Database connection pool initialized{ssl_info} (host: {self.host}, pool: 2-20 connections)")
        except Exception as e:
            print(f"✗ Failed to initialize database pool: {e}")
            raise
    
    def get_connection(self) -> PgConnection:
        """Get a connection from the pool with a health check"""
        if not self._initialized or self._pool is None:
            self.initialize()
        
        try:
            conn = self._pool.getconn()
            
            # Quick health check
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                # Connection is dead, try to get a new one
                print("Stale connection detected, replacing...")
                try:
                    self._pool.putconn(conn, close=True)
                except:
                    pass
                conn = self._pool.getconn()
                
            return conn
        except Exception as e:
            print(f"Error getting connection from pool: {e}")
            # If pool is full or broken, try re-initializing
            if "pool" in str(e).lower() or "exhausted" in str(e).lower():
                try:
                    self.close_all()
                    self.initialize()
                    return self._pool.getconn()
                except:
                    pass
            raise
    
    def release_connection(self, conn: PgConnection, error: bool = False):
        """Return a connection to the pool"""
        if self._pool is not None and conn is not None:
            try:
                self._pool.putconn(conn, close=error)
            except Exception as e:
                print(f"Warning: Failed to return connection to pool: {e}")
    
    def close_all(self):
        """Close all connections in the pool"""
        if self._pool is not None:
            try:
                self._pool.closeall()
                print("✓ Closed all database connections")
            except Exception as e:
                print(f"Warning: Error closing connection pool: {e}")
            finally:
                self._pool = None
                self._initialized = False


# Global database config instance
_db_config = DatabaseConfig()


def _is_embedding(value: list) -> bool:
    """Check if a list looks like an embedding vector (large float array)"""
    if not isinstance(value, list) or len(value) < 100:
        return False
    # Check if it's a list of floats (embedding vectors are typically 768, 1024, or 1536 floats)
    if len(value) in (768, 1024, 1536, 3072) and all(isinstance(x, (int, float)) for x in value[:10]):
        return True
    return False


def _serialize_value(value: Any) -> Any:
    """Serialize Python values for PostgreSQL.
    
    Handles:
    - Embeddings (large float arrays) → PostgreSQL vector format '[0.1, 0.2, ...]'
    - Regular lists → PostgreSQL arrays
    - Dicts → JSONB
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        # Check if this is an embedding vector (needs special formatting for pgvector)
        if _is_embedding(value):
            # Format as PostgreSQL vector string: '[0.1, 0.2, ...]'
            return '[' + ','.join(str(x) for x in value) + ']'
        # Regular arrays - pass directly to psycopg2
        return value
    if isinstance(value, dict):
        # Dicts become JSONB
        return json.dumps(value)
    return str(value)


def _deserialize_row(row: tuple, columns: List[str]) -> Dict[str, Any]:
    """Convert a row tuple to a dictionary with proper deserialization"""
    result = {}
    for i, col in enumerate(columns):
        value = row[i]
        # Handle JSON fields that come back as strings
        if isinstance(value, str) and col in ('embedding', 'metadata', 'result_data', 'job_data', 'app_metadata', 'user_metadata'):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
        result[col] = value
    return result


class QueryResult:
    """Result wrapper to match Supabase response format"""
    def __init__(self, data: List[Dict] = None, error: str = None, count: int = None):
        self.data = data or []
        self.error = error
        self.count = count if count is not None else len(self.data)


class QueryBuilder:
    """
    SQL Query builder that provides a Supabase-like interface for database operations.
    
    Supports: select, insert, update, delete, upsert operations with chainable filters.
    """
    
    def __init__(self, table_name: str, db_config: DatabaseConfig):
        self._table = table_name
        self._db_config = db_config
        self._operation = None
        self._columns = "*"
        self._data: Dict | List[Dict] = {}
        self._filters: List[tuple] = []
        self._or_filters: List[str] = []  # For OR conditions
        self._order_by: List[tuple] = []
        self._limit_val: int | None = None
        self._offset_val: int | None = None
        self._range_start: int | None = None
        self._range_end: int | None = None
        self._returning = True
        self._on_conflict = None
        self._ignore_duplicates = False  # For upsert ignore_duplicates
        self._negate_next = False  # For not_ support
        self._single = False  # For single() method
        self._count_mode = None  # For count="exact"
    
    def select(self, columns: str = "*", count: str = None) -> "QueryBuilder":
        """Select columns from the table
        
        Args:
            columns: Columns to select
            count: If "exact", also return total count (like Supabase count="exact")
        """
        self._operation = "SELECT"
        self._columns = columns
        self._count_mode = count  # "exact" means also count total rows
        return self
    
    def insert(self, data: Dict | List[Dict]) -> "QueryBuilder":
        """Insert data into the table"""
        self._operation = "INSERT"
        self._data = data if isinstance(data, list) else [data]
        return self
    
    def update(self, data: Dict) -> "QueryBuilder":
        """Update data in the table"""
        self._operation = "UPDATE"
        self._data = data
        return self
    
    def delete(self) -> "QueryBuilder":
        """Delete from the table"""
        self._operation = "DELETE"
        return self
    
    def upsert(self, data: Dict | List[Dict], on_conflict: str = "id", ignore_duplicates: bool = False) -> "QueryBuilder":
        """Upsert (insert or update on conflict)
        
        Args:
            data: Data to upsert
            on_conflict: Column(s) for conflict detection (can be comma-separated for multiple)
            ignore_duplicates: If True, do nothing on conflict instead of updating
        """
        self._operation = "UPSERT"
        self._data = data if isinstance(data, list) else [data]
        self._on_conflict = on_conflict
        self._ignore_duplicates = ignore_duplicates
        return self
    
    # Filter methods
    def eq(self, column: str, value: Any) -> "QueryBuilder":
        """Equal filter"""
        self._filters.append(("eq", column, value))
        return self
    
    def neq(self, column: str, value: Any) -> "QueryBuilder":
        """Not equal filter"""
        self._filters.append(("neq", column, value))
        return self
    
    def gt(self, column: str, value: Any) -> "QueryBuilder":
        """Greater than filter"""
        self._filters.append(("gt", column, value))
        return self
    
    def gte(self, column: str, value: Any) -> "QueryBuilder":
        """Greater than or equal filter"""
        self._filters.append(("gte", column, value))
        return self
    
    def lt(self, column: str, value: Any) -> "QueryBuilder":
        """Less than filter"""
        self._filters.append(("lt", column, value))
        return self
    
    def lte(self, column: str, value: Any) -> "QueryBuilder":
        """Less than or equal filter"""
        self._filters.append(("lte", column, value))
        return self
    
    def like(self, column: str, pattern: str) -> "QueryBuilder":
        """LIKE filter"""
        self._filters.append(("like", column, pattern))
        return self
    
    def ilike(self, column: str, pattern: str) -> "QueryBuilder":
        """Case-insensitive LIKE filter"""
        op = "not_ilike" if self._negate_next else "ilike"
        self._filters.append((op, column, pattern))
        self._negate_next = False
        return self
    
    def is_(self, column: str, value: Any) -> "QueryBuilder":
        """IS filter (for NULL checks)"""
        op = "is_not" if self._negate_next else "is"
        self._filters.append((op, column, value))
        self._negate_next = False
        return self
    
    def in_(self, column: str, values: List) -> "QueryBuilder":
        """IN filter"""
        if values:  # Only add filter if values is not empty
            self._filters.append(("in", column, values))
        return self
    
    def contains(self, column: str, value: Any) -> "QueryBuilder":
        """Contains filter for arrays/JSON"""
        self._filters.append(("contains", column, value))
        return self
    
    def or_(self, filter_string: str) -> "QueryBuilder":
        """
        OR filter using Supabase PostgREST syntax.
        Example: query.or_("deadline.is.null,deadline.gte.2024-01-01")
        """
        self._or_filters.append(filter_string)
        return self
    
    @property
    def not_(self) -> "QueryBuilder":
        """
        Negate the next filter.
        Example: query.not_.ilike("source", "%ted%")
        """
        self._negate_next = True
        return self
    
    # Ordering and pagination
    def order(self, column: str, desc: bool = False, nullsfirst: bool = False) -> "QueryBuilder":
        """Order results"""
        self._order_by.append((column, desc, nullsfirst))
        return self
    
    def limit(self, count: int) -> "QueryBuilder":
        """Limit results"""
        self._limit_val = count
        return self
    
    def offset(self, start: int) -> "QueryBuilder":
        """Offset results"""
        self._offset_val = start
        return self
    
    def range(self, start: int, end: int) -> "QueryBuilder":
        """Range pagination (like Supabase)"""
        self._range_start = start
        self._range_end = end
        return self
    
    def single(self) -> "QueryBuilder":
        """
        Return a single row instead of an array.
        Sets limit to 1 and returns data as object instead of array.
        """
        self._limit_val = 1
        self._single = True
        return self
    
    def maybeSingle(self) -> "QueryBuilder":
        """
        Return a single row or None if not found.
        Like single() but doesn't error on no results.
        """
        return self.single()
    
    def _build_where_clause(self) -> tuple[str, List]:
        """Build WHERE clause from filters"""
        if not self._filters and not self._or_filters:
            return "", []
        
        conditions = []
        params = []
        
        for op, column, value in self._filters:
            if op == "eq":
                conditions.append(f'"{column}" = %s')
                params.append(_serialize_value(value))
            elif op == "neq":
                conditions.append(f'"{column}" != %s')
                params.append(_serialize_value(value))
            elif op == "gt":
                conditions.append(f'"{column}" > %s')
                params.append(_serialize_value(value))
            elif op == "gte":
                conditions.append(f'"{column}" >= %s')
                params.append(_serialize_value(value))
            elif op == "lt":
                conditions.append(f'"{column}" < %s')
                params.append(_serialize_value(value))
            elif op == "lte":
                conditions.append(f'"{column}" <= %s')
                params.append(_serialize_value(value))
            elif op == "like":
                conditions.append(f'"{column}" LIKE %s')
                params.append(value)
            elif op == "ilike":
                conditions.append(f'"{column}" ILIKE %s')
                params.append(value)
            elif op == "not_ilike":
                conditions.append(f'"{column}" NOT ILIKE %s')
                params.append(value)
            elif op == "is":
                if value is None or str(value).lower() == "null":
                    conditions.append(f'"{column}" IS NULL')
                else:
                    conditions.append(f'"{column}" IS %s')
                    params.append(value)
            elif op == "is_not":
                if value is None or str(value).lower() == "null":
                    conditions.append(f'"{column}" IS NOT NULL')
                else:
                    conditions.append(f'"{column}" IS NOT %s')
                    params.append(value)
            elif op == "in":
                if value:
                    placeholders = ", ".join(["%s"] * len(value))
                    conditions.append(f'"{column}" IN ({placeholders})')
                    params.extend([_serialize_value(v) for v in value])
            elif op == "contains":
                conditions.append(f'"{column}" @> %s')
                params.append(_serialize_value(value))
        
        # Handle OR filters (Supabase PostgREST syntax: "col.op.value,col.op.value")
        for or_filter in self._or_filters:
            or_parts = []
            for part in or_filter.split(","):
                part = part.strip()
                if ".is.null" in part:
                    col = part.split(".is.null")[0]
                    or_parts.append(f'"{col}" IS NULL')
                elif ".gte." in part:
                    col, val = part.split(".gte.")
                    or_parts.append(f'"{col}" >= %s')
                    params.append(val)
                elif ".gt." in part:
                    col, val = part.split(".gt.")
                    or_parts.append(f'"{col}" > %s')
                    params.append(val)
                elif ".lte." in part:
                    col, val = part.split(".lte.")
                    or_parts.append(f'"{col}" <= %s')
                    params.append(val)
                elif ".lt." in part:
                    col, val = part.split(".lt.")
                    or_parts.append(f'"{col}" < %s')
                    params.append(val)
                elif ".eq." in part:
                    col, val = part.split(".eq.")
                    or_parts.append(f'"{col}" = %s')
                    params.append(val)
                elif ".neq." in part:
                    col, val = part.split(".neq.")
                    or_parts.append(f'"{col}" != %s')
                    params.append(val)
            if or_parts:
                conditions.append(f'({" OR ".join(or_parts)})')
        
        if not conditions:
            return "", []
        
        return " WHERE " + " AND ".join(conditions), params
    
    def _build_order_clause(self) -> str:
        """Build ORDER BY clause"""
        if not self._order_by:
            return ""
        
        parts = []
        for column, desc, nullsfirst in self._order_by:
            part = f'"{column}"'
            if desc:
                part += " DESC"
            else:
                part += " ASC"
            if nullsfirst:
                part += " NULLS FIRST"
            parts.append(part)
        
        return " ORDER BY " + ", ".join(parts)
    
    def _build_pagination(self) -> str:
        """Build LIMIT/OFFSET clause"""
        parts = []
        
        if self._range_start is not None and self._range_end is not None:
            limit = self._range_end - self._range_start + 1
            parts.append(f" LIMIT {limit} OFFSET {self._range_start}")
        else:
            if self._limit_val is not None:
                parts.append(f" LIMIT {self._limit_val}")
            if self._offset_val is not None:
                parts.append(f" OFFSET {self._offset_val}")
        
        return "".join(parts)
    
    def execute(self, max_retries: int = 3, retry_delay: float = 0.5) -> QueryResult:
        """Execute the query with retry logic"""
        last_error = None
        
        for attempt in range(max_retries):
            conn = None
            try:
                conn = self._db_config.get_connection()
                
                with conn.cursor() as cursor:
                    if self._operation == "SELECT":
                        result = self._execute_select(cursor)
                    elif self._operation == "INSERT":
                        result = self._execute_insert(cursor, conn)
                    elif self._operation == "UPDATE":
                        result = self._execute_update(cursor, conn)
                    elif self._operation == "DELETE":
                        result = self._execute_delete(cursor, conn)
                    elif self._operation == "UPSERT":
                        result = self._execute_upsert(cursor, conn)
                    else:
                        raise ValueError(f"Unknown operation: {self._operation}")
                    
                    # Handle single() - return first item as object instead of array
                    if self._single and result.data:
                        result = QueryResult(data=result.data[0], error=result.error, count=1)
                    elif self._single and not result.data:
                        result = QueryResult(data=None, error=result.error, count=0)
                    
                    return result
                    
            except (psycopg2.OperationalError, psycopg2.InterfaceError, ConnectionError) as e:
                last_error = e
                print(f"Database connection error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    # Close connection if it failed
                    if conn:
                        try:
                            self._db_config.release_connection(conn, error=True)
                            conn = None
                        except:
                            pass
            except Exception as e:
                # Non-connection errors should not retry
                print(f"Database query error: {e}")
                traceback.print_exc()
                if conn:
                    self._db_config.release_connection(conn)
                    conn = None
                return QueryResult(data=[] if not self._single else None, error=str(e))
            finally:
                if conn:
                    try:
                        self._db_config.release_connection(conn)
                    except:
                        pass
        
        # All retries failed
        return QueryResult(data=[] if not self._single else None, error=str(last_error) if last_error else "Unknown error")
    
    def _execute_select(self, cursor) -> QueryResult:
        """Execute SELECT query"""
        import re
        
        # Handle column selection - expand * or parse column list
        # Handle joins like "id, name, client_rfps(original_rfp_date)"
        columns = self._columns
        
        # Check for Supabase-style joins like "client_rfps(original_rfp_date)"
        # We'll strip these for now as they require special JOIN handling
        columns = re.sub(r',?\s*\w+\([^)]+\)', '', columns).strip().rstrip(',')
        if not columns:
            columns = "*"
        
        # Build WHERE clause
        where_clause, params = self._build_where_clause()
        
        # Get total count if count mode is "exact"
        total_count = None
        if self._count_mode == "exact":
            count_query = f'SELECT COUNT(*) FROM "{self._table}"' + where_clause
            cursor.execute(count_query, params)
            total_count = cursor.fetchone()[0]
        
        # Main query
        query = f'SELECT {columns} FROM "{self._table}"'
        query += where_clause
        query += self._build_order_clause()
        query += self._build_pagination()
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Get column names from cursor description
        if cursor.description:
            col_names = [desc[0] for desc in cursor.description]
            data = [_deserialize_row(row, col_names) for row in rows]
        else:
            data = []
        
        return QueryResult(data=data, count=total_count if total_count is not None else len(data))
    
    def _execute_insert(self, cursor, conn) -> QueryResult:
        """Execute INSERT query"""
        if not self._data:
            return QueryResult(data=[])
        
        # Get column names from first record
        columns = list(self._data[0].keys())
        col_names = ", ".join([f'"{c}"' for c in columns])
        placeholders = ", ".join(["%s"] * len(columns))
        
        query = f'INSERT INTO "{self._table}" ({col_names}) VALUES ({placeholders})'
        if self._returning:
            query += " RETURNING *"
        
        results = []
        for record in self._data:
            values = [_serialize_value(record.get(c)) for c in columns]
            cursor.execute(query, values)
            if self._returning and cursor.description:
                row = cursor.fetchone()
                if row:
                    col_names_result = [desc[0] for desc in cursor.description]
                    results.append(_deserialize_row(row, col_names_result))
        
        conn.commit()
        return QueryResult(data=results)
    
    def _execute_update(self, cursor, conn) -> QueryResult:
        """Execute UPDATE query"""
        if not self._data:
            return QueryResult(data=[])
        
        set_parts = []
        params = []
        for col, value in self._data.items():
            set_parts.append(f'"{col}" = %s')
            params.append(_serialize_value(value))
        
        query = f'UPDATE "{self._table}" SET {", ".join(set_parts)}'
        where_clause, where_params = self._build_where_clause()
        query += where_clause
        params.extend(where_params)
        
        if self._returning:
            query += " RETURNING *"
        
        cursor.execute(query, params)
        
        results = []
        if self._returning and cursor.description:
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description]
            results = [_deserialize_row(row, col_names) for row in rows]
        
        conn.commit()
        return QueryResult(data=results)
    
    def _execute_delete(self, cursor, conn) -> QueryResult:
        """Execute DELETE query"""
        query = f'DELETE FROM "{self._table}"'
        where_clause, params = self._build_where_clause()
        query += where_clause
        
        if self._returning:
            query += " RETURNING *"
        
        cursor.execute(query, params)
        
        results = []
        if self._returning and cursor.description:
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description]
            results = [_deserialize_row(row, col_names) for row in rows]
        
        conn.commit()
        return QueryResult(data=results)
    
    def _execute_upsert(self, cursor, conn) -> QueryResult:
        """Execute UPSERT (INSERT ... ON CONFLICT) query"""
        if not self._data:
            return QueryResult(data=[])
        
        columns = list(self._data[0].keys())
        col_names = ", ".join([f'"{c}"' for c in columns])
        placeholders = ", ".join(["%s"] * len(columns))
        
        # Handle multiple conflict columns (comma-separated)
        conflict_cols = [c.strip() for c in self._on_conflict.split(",")]
        conflict_clause = ", ".join([f'"{c}"' for c in conflict_cols])
        
        # Build ON CONFLICT clause
        if self._ignore_duplicates:
            # DO NOTHING on conflict
            conflict_action = "DO NOTHING"
        else:
            # DO UPDATE - exclude conflict columns from update
            update_cols = [c for c in columns if c not in conflict_cols]
            if update_cols:
                update_parts = [f'"{c}" = EXCLUDED."{c}"' for c in update_cols]
                conflict_action = f"DO UPDATE SET {', '.join(update_parts)}"
            else:
                conflict_action = "DO NOTHING"
        
        query = f'''
            INSERT INTO "{self._table}" ({col_names}) 
            VALUES ({placeholders})
            ON CONFLICT ({conflict_clause}) 
            {conflict_action}
        '''
        if self._returning:
            query += " RETURNING *"
        
        results = []
        for record in self._data:
            values = [_serialize_value(record.get(c)) for c in columns]
            cursor.execute(query, values)
            if self._returning and cursor.description:
                row = cursor.fetchone()
                if row:
                    col_names_result = [desc[0] for desc in cursor.description]
                    results.append(_deserialize_row(row, col_names_result))
        
        conn.commit()
        return QueryResult(data=results)


class RPCBuilder:
    """Builder for calling PostgreSQL stored procedures/functions (RPC)"""
    
    def __init__(self, function_name: str, params: Dict, db_config: DatabaseConfig):
        self._function_name = function_name
        self._params = params
        self._db_config = db_config
    
    def execute(self, max_retries: int = 3, retry_delay: float = 0.5) -> QueryResult:
        """Execute the RPC call"""
        last_error = None
        
        for attempt in range(max_retries):
            conn = None
            try:
                conn = self._db_config.get_connection()
                
                with conn.cursor() as cursor:
                    # Build parameter list for function call
                    param_names = list(self._params.keys())
                    param_values = [_serialize_value(self._params[k]) for k in param_names]
                    
                    # Build the function call
                    placeholders = ", ".join([f"{name} := %s" for name in param_names])
                    query = f'SELECT * FROM "{self._function_name}"({placeholders})'
                    
                    cursor.execute(query, param_values)
                    rows = cursor.fetchall()
                    
                    # Get column names
                    if cursor.description:
                        col_names = [desc[0] for desc in cursor.description]
                        data = [_deserialize_row(row, col_names) for row in rows]
                    else:
                        data = []
                    
                    return QueryResult(data=data)
                    
            except (psycopg2.OperationalError, psycopg2.InterfaceError, ConnectionError) as e:
                last_error = e
                print(f"Database RPC error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    if conn:
                        try:
                            self._db_config.release_connection(conn, error=True)
                            conn = None
                        except:
                            pass
            except Exception as e:
                print(f"RPC error calling {self._function_name}: {e}")
                traceback.print_exc()
                if conn:
                    self._db_config.release_connection(conn)
                    conn = None
                return QueryResult(data=[], error=str(e))
            finally:
                if conn:
                    try:
                        self._db_config.release_connection(conn)
                    except:
                        pass
        
        return QueryResult(data=[], error=str(last_error) if last_error else "Unknown error")


class Database:
    """
    Main database interface providing Supabase-like API for PostgreSQL operations.
    
    Usage:
        db = Database()
        
        # Initialize (auto-called on first use if DATABASE_URL is set)
        db.initialize(database_url="postgresql://...")
        
        # Query operations
        result = db.table("users").select("*").eq("status", "active").execute()
        
        # RPC calls
        result = db.rpc("match_questions", {"embedding": [...], "count": 5}).execute()
    """
    
    def __init__(self):
        self._config = _db_config
        self._initialized = False
    
    def initialize(self, database_url: str = None, **kwargs):
        """Initialize the database connection pool"""
        import os
        url = database_url or os.getenv("DATABASE_URL")
        self._config.initialize(database_url=url, **kwargs)
        self._initialized = True
    
    def _ensure_initialized(self):
        """Ensure database is initialized"""
        if not self._initialized:
            self.initialize()
    
    def table(self, table_name: str) -> QueryBuilder:
        """Start a query on a table"""
        self._ensure_initialized()
        return QueryBuilder(table_name, self._config)
    
    def rpc(self, function_name: str, params: Dict = None) -> RPCBuilder:
        """Call a PostgreSQL function (RPC)"""
        self._ensure_initialized()
        return RPCBuilder(function_name, params or {}, self._config)
    
    def execute_raw(self, query: str, params: tuple = None, fetch: bool = True) -> QueryResult:
        """Execute a raw SQL query"""
        self._ensure_initialized()
        conn = None
        try:
            conn = self._config.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                
                if fetch and cursor.description:
                    rows = cursor.fetchall()
                    col_names = [desc[0] for desc in cursor.description]
                    data = [_deserialize_row(row, col_names) for row in rows]
                else:
                    conn.commit()
                    data = []
                
                return QueryResult(data=data)
        except Exception as e:
            print(f"Raw query error: {e}")
            traceback.print_exc()
            if conn:
                self._config.release_connection(conn, error=True)
                conn = None
            return QueryResult(data=[], error=str(e))
        finally:
            if conn:
                self._config.release_connection(conn)
    
    def close(self):
        """Close all database connections"""
        self._config.close_all()
        self._initialized = False
    
    def test_connection(self) -> bool:
        """Test the database connection"""
        try:
            self._ensure_initialized()
            result = self.execute_raw("SELECT NOW() as current_time")
            if result.data:
                print(f"✓ Database connection successful! Server time: {result.data[0]['current_time']}")
                return True
            return False
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            return False


# Global database instance
db = Database()


def get_db() -> Database:
    """Get the global database instance"""
    return db


def init_db(database_url: str = None):
    """Initialize the global database instance"""
    db.initialize(database_url=database_url)
    return db


