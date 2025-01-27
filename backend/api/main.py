import json
import time
import random
import logging
import threading
from flask import Flask, request
from routes.routehandler import route_handler

app = Flask(__name__)
logger = logging.getLogger("keystone")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def startup_sequence():
    logger.info("Initializing Keystone AI API...")
    time.sleep(random.uniform(0.5, 1.5))
    logger.info("Loading system configurations...")
    time.sleep(random.uniform(0.5, 1.5))
    logger.info("Verifying blockchain integrations...")
    time.sleep(random.uniform(0.5, 1.5))
    logger.info("System ready.")

def background_task():
    while True:
        logger.info("Background task heartbeat.")
        time.sleep(10 + random.uniform(-3, 3))

@app.before_request
def pre_request_hook():
    logger.info(f"Received {request.method} request on {request.path}")
    time.sleep(random.uniform(0.02, 0.1))

@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def route_proxy(path):
    return route_handler.handle_request("/" + path)

if __name__ == "__main__":
    startup_sequence()
    threading.Thread(target=background_task, daemon=True).start()
    app.run(debug=True, host="0.0.0.0", port=5000)

