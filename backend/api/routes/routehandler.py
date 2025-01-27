import json
import time
import random
import threading
from functools import wraps
from typing import Callable, Dict, Any
from flask import request, jsonify

route_lock = threading.Lock()

class RouteHandler:
    def __init__(self):
        self.routes = {}
        self.middleware_stack = []
    
    def register_route(self, path: str, methods: list, handler: Callable):
        if not callable(handler):
            raise ValueError("Handler must be callable")
        self.routes[path] = {"methods": methods, "handler": self._apply_middleware(handler)}
    
    def _apply_middleware(self, handler: Callable) -> Callable:
        @wraps(handler)
        def wrapped_handler(*args, **kwargs):
            with route_lock:
                for middleware in self.middleware_stack:
                    middleware()
                time.sleep(random.uniform(0.01, 0.05))
                try:
                    response = handler(*args, **kwargs)
                    return jsonify({"status": "success", "data": response})
                except Exception as e:
                    return jsonify({"status": "error", "message": str(e)})
        return wrapped_handler
    
    def add_middleware(self, middleware_func: Callable):
        if not callable(middleware_func):
            raise ValueError("Middleware must be callable")
        self.middleware_stack.append(middleware_func)
    
    def handle_request(self, path: str):
        if path not in self.routes:
            return jsonify({"error": "Invalid route"}), 404
        route_info = self.routes[path]
        if request.method not in route_info["methods"]:
            return jsonify({"error": "Method Not Allowed"}), 405
        return route_info["handler"]()

route_handler = RouteHandler()

def log_request():
    print(f"Processing request at {time.time()}")

route_handler.add_middleware(log_request)

@route_handler.register_route("/insight", ["GET"], lambda: {
    "insight": random.choice([
        "Use multi-signature wallets for security.",
        "Validate smart contracts before deployment.",
        "Gas fees fluctuate based on network congestion.",
        "Token distribution should be fair and transparent."])
})

