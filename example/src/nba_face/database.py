"""Face identity storage and batched pgvector searches."""
import math
import os

import psycopg
from psycopg.rows import dict_row


class FaceSearch:
    def __init__(self, config):
        # Prefer a registry URL; retain environment-based configuration support.
        dsn_env = config.get("dsn_env", "NBA_FACE_DATABASE_URL")
        self.dsn = config.get("url") or os.environ.get(dsn_env)
        self.options = {
            target: config[source]
            for source, target in (("name", "dbname"), ("username", "user"),
                                   ("password", "password"), ("host", "host"), ("port", "port"))
            if source in config
        }
        if not self.dsn and not self.options:
            raise ValueError(
                "Configure config.db.url with a PostgreSQL connection URL, "
                "or configure config.db.name, username, password, host, and port. "
                f"Alternatively, set environment variable {dsn_env!r} "
                "(named by config.db.dsn_env) to a PostgreSQL DSN."
            )
        self.timeout = int(config.get("statement_timeout_ms", 5000))
        if self.timeout <= 0:
            raise ValueError("db.statement_timeout_ms must be positive")
        self.connection = None

    def _connect(self):
        if self.connection is None or self.connection.closed or self.connection.broken:
            self.connection = psycopg.connect(
                self.dsn or "", **self.options, autocommit=True, row_factory=dict_row,
                connect_timeout=10, options=f"-c statement_timeout={self.timeout}",
            )
        return self.connection

    @staticmethod
    def _vector(embedding):
        if len(embedding) != 512 or not all(math.isfinite(v) for v in embedding):
            raise ValueError("PostgreSQL identity storage requires finite 512D embeddings")
        return "[" + ",".join(map(str, embedding)) + "]"

    def register(self, employee_id, embedding, image_url):
        with self._connect().cursor() as cursor:
            cursor.execute("""
                INSERT INTO public.identity (employee_id, embedding, image_url, type)
                VALUES (%s, %s::vector, %s, 'front')
                RETURNING id AS identity_id, employee_id, image_url
            """, (employee_id, self._vector(embedding), image_url))
            return dict(cursor.fetchone())

    def delete(self, identity_id):
        with self._connect().cursor() as cursor:
            cursor.execute("""
                UPDATE public.identity
                SET deleted_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND deleted_at IS NULL
                RETURNING id
            """, (identity_id,))
            return {"identity_id": identity_id, "deleted": cursor.fetchone() is not None}

    def search_batch(self, embeddings):
        """One round trip; preserve order and include an unmatched row per input."""
        if not embeddings:
            return []
        vectors = [self._vector(embedding) for embedding in embeddings]
        with self._connect().cursor() as cursor:
            cursor.execute("""
                SELECT query.ordinality, match.identity_id, match.employee_id,
                       match.image_url, match.pose, match.similarity
                FROM unnest(%s::text[]) WITH ORDINALITY AS query(embedding, ordinality)
                LEFT JOIN LATERAL (
                    SELECT i.id AS identity_id, i.employee_id, i.image_url, i.type AS pose,
                           1 - (i.embedding <=> query.embedding::vector) AS similarity
                    FROM public.identity i
                    WHERE i.deleted_at IS NULL AND i.embedding IS NOT NULL
                    ORDER BY i.embedding <=> query.embedding::vector
                    LIMIT 1
                ) AS match ON TRUE
                ORDER BY query.ordinality
            """, (vectors,))
            return [dict(row) for row in cursor.fetchall()]

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
