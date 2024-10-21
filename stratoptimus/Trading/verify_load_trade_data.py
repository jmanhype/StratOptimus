import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)

from scripts.optimized.Trading.backtester_module import load_trade_data

def main():
    filename = "big_optimize_1016.pkl"
    trading_data = load_trade_data(filename)
    
    if trading_data:
        logger.info(f"Successfully loaded trading data from {filename}")
        logger.info(f"Number of assets loaded: {len(trading_data)}")
        # Optionally, print the first asset's data
        first_asset = list(trading_data.keys())[0]
        logger.info(f"First asset: {first_asset}")
        logger.info(f"Data preview:\n{trading_data[first_asset].head()}")
    else:
        logger.error("Trading data is empty. Please check the pickle file.")

if __name__ == "__main__":
    main()
