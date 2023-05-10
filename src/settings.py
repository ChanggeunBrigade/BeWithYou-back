from dotenv import dotenv_values

from flask import Flask, g

config = dotenv_values(".env")

app = Flask(__name__)


@app.teardown_appcontext
def teardown_db(exception):
    cur = g.pop("cur", None)
    conn = g.pop("db", None)

    try:
        cur.close()
    except:
        pass

    try:
        conn.close()
    except:
        pass


class SingletonMeta(type):
    _instances = {}

    def __new__(cls, name, bases, attrs):
        if name in cls._instances:
            return cls._instances[name]
        else:
            instance = super().__new__(cls, name, bases, attrs)
            cls._instances[name] = instance
            return instance
