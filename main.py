from flask import Flask, request, jsonify
app = Flask(__name__)

@app.get("/")
def health():
    return "OK"

@app.post("/ingest")
def ingest():
    payload = request.get_json()
    # TODO: write to BigQuery
    return jsonify(status="received")
