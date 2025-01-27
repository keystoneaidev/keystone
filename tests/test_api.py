import unittest
import json
from backend.api.main import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    def test_home_route(self):
        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("message", data)
        self.assertEqual(data["message"], "Welcome to Keystone AI")
    
    def test_recommendations_route(self):
        response = self.app.get("/recommendations")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("insights", data)
        self.assertIsInstance(data["insights"], str)
    
    def test_invalid_route(self):
        response = self.app.get("/invalid")
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertIn("error", data)

if __name__ == "__main__":
    unittest.main()

