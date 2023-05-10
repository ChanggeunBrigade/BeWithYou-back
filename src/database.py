import psycopg2
import psycopg2.extensions

from src.settings import config, SingletonMeta


class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.conn: psycopg2.extensions.connection = psycopg2.connect(
            host=config["PSQL_HOST"],
            port=config["PSQL_PORT"],
            database="data",
            user=config["PSQL_USER"],
            password=config["PSQL_PASS"],
        )

    def get_table_data(self, table_name: str) -> list:
        with self.conn.cursor() as cursor:
            cursor.execute("select * from {}".format(table_name))
            data = cursor.fetchall()
        return data
