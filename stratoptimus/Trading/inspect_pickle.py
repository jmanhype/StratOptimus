import os
import pickle
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def inspect_pickle(filename: str = '/home/batmanosama/poc-kagnar/experiments/dslmodel-prefect/MetaGPT-MathAI/examples/aflow/scripts/optimized/Trading/big_optimize_1016.pkl'):
    """
    Load and inspect the contents of a pickle file containing trading data.

    Args:
        filename (str): Full path to the pickle file.
    """
    try:
        logger.info(f"Attempting to load pickle file from: {filename}")

        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                loaded_trade_data_dict = pickle.load(f)
            
            logger.info(f"Successfully loaded trade data from {filename}")
            logger.info(f"Number of assets loaded: {len(loaded_trade_data_dict)}")
            
            # Print information about the loaded data
            for asset, df in loaded_trade_data_dict.items():
                logger.info(f"\nAsset: {asset}")
                logger.info(f"DataFrame shape: {df.shape}")
                logger.info(f"Columns: {df.columns.tolist()}")
                logger.info(f"Data types:\n{df.dtypes}")
                logger.info(f"Sample data (first 5 rows):\n{df.head()}")
                logger.info(f"Sample data (last 5 rows):\n{df.tail()}")
                logger.info("-" * 50)

                # Break after inspecting a few assets to avoid overwhelming output
                if list(loaded_trade_data_dict.keys()).index(asset) >= 4:
                    logger.info("Showing only the first 5 assets to limit output.")
                    break

        else:
            logger.error(f"Pickle file not found at {filename}")
    except Exception as e:
        logger.exception(f"Error inspecting trade data from {filename}: {e}")

if __name__ == "__main__":
    inspect_pickle()