import json
import time
import random
import requests
import logging
from web3 import Web3
from typing import Dict, Any
from config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fetch_data")

class BlockchainDataFetcher:
    def __init__(self, provider_url: str):
        self.w3 = Web3(Web3.HTTPProvider(provider_url))
        if not self.w3.is_connected():
            raise ConnectionError("Unable to connect to blockchain provider")
        logger.info("Connected to blockchain provider.")

    def fetch_latest_block(self) -> Dict[str, Any]:
        block = self.w3.eth.get_block("latest")
        return {
            "block_number": block.number,
            "hash": block.hash.hex(),
            "transactions": len(block.transactions),
            "timestamp": block.timestamp
        }

    def fetch_transaction(self, tx_hash: str) -> Dict[str, Any]:
        tx = self.w3.eth.get_transaction(tx_hash)
        return {
            "hash": tx.hash.hex(),
            "from": tx["from"],
            "to": tx["to"],
            "value": tx["value"]
        }

class MarketDataFetcher:
    def __init__(self, api_url: str):
        self.api_url = api_url

    def fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        url = f"{self.api_url}/ticker?symbol={symbol}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to fetch market data for {symbol}")
            return {}

if __name__ == "__main__":
    blockchain_fetcher = BlockchainDataFetcher(Config.BLOCKCHAIN_PROVIDER)
    market_fetcher = MarketDataFetcher("https://api.coingecko.com/api/v3")
    
    latest_block = blockchain_fetcher.fetch_latest_block()
    logger.info(f"Latest Block: {json.dumps(latest_block, indent=2)}")
    
    market_data = market_fetcher.fetch_market_data("bitcoin")
    logger.info(f"Bitcoin Market Data: {json.dumps(market_data, indent=2)}")

