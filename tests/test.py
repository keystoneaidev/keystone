import unittest
import json
import random
from scripts.fetch_data import BlockchainDataFetcher, MarketDataFetcher
from config import Config

class TestBlockchainDataFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = BlockchainDataFetcher(Config.BLOCKCHAIN_PROVIDER)
    
    def test_fetch_latest_block(self):
        block_data = self.fetcher.fetch_latest_block()
        self.assertIn("block_number", block_data)
        self.assertIn("hash", block_data)
        self.assertIn("transactions", block_data)
        self.assertIn("timestamp", block_data)

    def test_fetch_transaction(self):
        fake_tx_hash = "0x" + "1" * 64  # Placeholder hash for test
        with self.assertRaises(Exception):  # Expect failure with fake hash
            self.fetcher.fetch_transaction(fake_tx_hash)

class TestMarketDataFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = MarketDataFetcher("https://api.coingecko.com/api/v3")
    
    def test_fetch_market_data(self):
        market_data = self.fetcher.fetch_market_data("bitcoin")
        self.assertIsInstance(market_data, dict)
        self.assertIn("market_data", market_data)  # Ensure key exists

if __name__ == "__main__":
    unittest.main()

