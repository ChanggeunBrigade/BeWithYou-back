import psycopg2
import psycopg2.extensions
from psycopg2 import sql

from settings import config, SingletonMeta


class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.conn: psycopg2.extensions.connection = psycopg2.connect(
            host=config["PSQL_HOST"],
            port=config["PSQL_PORT"],
            database="data",
            user=config["PSQL_USER"],
            password=config["PSQL_PASS"],
        )

    def __del__(self):
        try:
            self.conn.close()
        except:
            pass

    def get_table_data(self, table_name: str) -> list:
        with self.conn.cursor() as cursor:
            cursor.execute(sql.SQL("select * from {}").format(sql.Identifier(table_name)))
            data = cursor.fetchall()
        return data

    def insert_label(self, ts, value):
        cursor: psycopg2.extensions.cursor = self.conn.cursor()
        cursor.execute(
            "insert into label (time, label) values (%s, %s) on conflict (time) do update set label = %s",
            (ts, value, value),
        )
        self.conn.commit()
        cursor.close()
