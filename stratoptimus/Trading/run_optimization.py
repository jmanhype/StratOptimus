import asyncio
import logging
from typing import Dict, Any
import os
import json
from datetime import datetime
import sys
import vectorbtpro as vbt
from metagpt.configs.llm_config import LLMConfig
from metagpt.configs.llm_config import LLMConfig
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, project_root)
from scripts.config_utils import load_config
from metagpt.provider.llm_provider_registry import create_llm_instance
from scripts.optimized.Trading.recursive_optimization import recursive_optimization
from scripts.optimized.Trading.backtester_module import load_trade_data  # Corrected import path
from scripts.optimized.Trading.backtester_module import load_trade_data  # Corrected import path
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

async def main():
    try:
        # Load configuration
        config = load_config(os.environ.get("TRADING_CONFIG_PATH", "config/trading_config.yaml"))
        
        # Create LLM instance
        llm_config = LLMConfig(**config["llm_config"])
        llm = create_llm_instance(llm_config)

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output/trading_optimization_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        dataset = config.get("dataset", [])  # Assuming dataset is defined in the config
        
        # Define initial parameters
        initial_params = config.get("initial_params", {
            "take_profit": 0.08,
            "stop_loss": 0.12,
            "sl_window": 400,
            "max_orders": 3,
            "post_buy_delay": 2,
            "post_sell_delay": 5,
            "macd_signal_fast": 120,
            "macd_signal_slow": 260,
            "macd_signal_signal": 90,
            "min_macd_signal_threshold": 0.0,
            "max_macd_signal_threshold": 0.0,
            "enable_sl_mod": False,
            "enable_tp_mod": False,
            "ma_window": 20,
            "fees": 0.001  # Add this line
        })
        
        # Load or fetch trading data using the backtester's load_trade_data function
        trading_data = load_trade_data("big_optimize_1016.pkl")

        if trading_data is None:
            logger.error("Failed to load trading data. Please ensure the pickle file exists and is accessible.")
            return

        # Create a dataset with a subset of assets for optimization
        dataset = list(trading_data.keys())[:5]  # Use the first 5 assets for example
        
        # Run recursive optimization
        best_strategy, best_performance = await recursive_optimization(
            initial_params=initial_params,
            dataset=dataset,
            llm=llm,
            trading_data=trading_data,
            max_iterations=config.get("max_rounds", 45),
            convergence_threshold=config.get("convergence_threshold", 0.01),
            parallel_evaluations=config.get("parallel_evaluations", 5)  # Added parallel_evaluations parameter
        )

        # Save the best strategy
        if best_strategy:
            with open(os.path.join(output_dir, "best_trading_strategy_final.json"), "w") as f:
                json.dump(best_strategy, f, indent=2)
            logger.info(f"Optimization completed. Best strategy saved to {output_dir}")
            logger.info(f"Best performance: {best_performance}")
        else:
            logger.error("Optimization failed.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
