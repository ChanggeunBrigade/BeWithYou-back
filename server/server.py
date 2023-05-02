from flask import Flask, g, jsonify
import psycopg2
from dotenv import dotenv_values

config = dotenv_values("../.env")

app = Flask(__name__)


def get_db_cursor():
    if "db" not in g:
        g.db = psycopg2.connect(
            host="localhost",
            port=5432,
            database="data",
            user=config["PSQL_USER"],
            password=config["PSQL_PASS"],
        )
        g.cur = g.db.cursor()

    return g.cur


@app.teardown_appcontext
def teardown_db(exception):
    cur = g.pop("cur", None)
    db = g.pop("db", None)

    try:
        cur.close()
    except:
        pass

    try:
        db.close()
    except:
        pass


@app.route("/video")
def video():
    cursor = get_db_cursor()
    cursor.execute("select * from opencv")
    video = cursor.fetchall()
    return jsonify(video)


@app.route("/audio")
def audio():
    cursor = get_db_cursor()
    cursor.execute("select * from audio")
    audio = cursor.fetchall()
    return jsonify(audio)


@app.route("/csi")
def csi():
    cursor = get_db_cursor()
    cursor.execute("select * from tcpdump")
    csi = cursor.fetchall()
    return jsonify(csi)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
