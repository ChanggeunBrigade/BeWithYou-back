from flask import jsonify

import database
from settings import app

db = database.Database()


@app.route("/video")
def video():
    data = db.get_table_data("opencv")
    return jsonify(data)


@app.route("/audio")
def audio():
    data = db.get_table_data("audio")
    return jsonify(data)


@app.route("/csi")
def csi():
    data = db.get_table_data("tcpdump")
    return jsonify(data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
