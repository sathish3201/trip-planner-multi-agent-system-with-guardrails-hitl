
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from langgraph.checkpoint.postgres import PostgresSaver

from app.config import DATABASE_URL


def create_checkpointer():

    database_url = DATABASE_URL

    if "sslmode=" not in database_url.lower():

        separator = "&" if "?" in database_url else "?"

        database_url += f"{separator}sslmode=require"

    pool = ConnectionPool(
        conninfo=database_url,
        max_size=20,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
        },
    )

    checkpointer = PostgresSaver(pool)

    checkpointer.setup()

    return pool, checkpointer
