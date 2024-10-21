# -*- coding: utf-8 -*-
# @Date    : 10/19/2024
# @Author  : Your Name
# @Desc    : Script to apply and review optimized trading strategies

import json
import argparse
import logging
from typing import List, Dict, Any
from scripts.optimized.Trading.workflows.round_1.graph import TradingWorkflow
from metagpt.provider.llm_provider_registry import create_llm_instance

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_strategy(strategy_path: str) -> List[Dict[str, Any]]:
    """
    Load the optimized trading strategy from a JSON file.
    
    Args:
        strategy_path (str): Path to the JSON file containing the trading strategies.
    
    Returns:
        List[Dict[str, Any]]: Loaded strategy data.
    """
    try:
        with open(strategy_path, 'r') as f:
            strategies = json.load(f)
        logger.info(f"Successfully loaded strategy from {strategy_path}")
        return strategies
    except Exception as e:
        logger.error(f"Error loading strategy from {strategy_path}: {str(e)}")
        raise

def print_strategy(strategy_id: str, evaluation: Dict[str, Any]):
    """
    Print the details of a trading strategy evaluation.
    
    Args:
        strategy_id (str): The identifier of the trading strategy.
        evaluation (Dict[str, Any]): Evaluation results for the trading strategy.
    """
    print(f"Strategy ID: {strategy_id}")
    print(f"Total Return: {evaluation.get('total_return', 0.0):.2f}%")
    print(f"Sharpe Ratio: {evaluation.get('sharpe_ratio', 0.0):.2f}")
    print(f"Maximum Drawdown: {evaluation.get('max_drawdown', 0.0):.2f}%")
    print(f"Win Rate: {evaluation.get('win_rate', 0.0):.2f}%")
    print(f"Profit Factor: {evaluation.get('profit_factor', 0.0):.2f}")
    if 'strategy_details' in evaluation:
        print(f"Strategy Details:\n{evaluation['strategy_details']}")
    print("-" * 50)

async def generate_new_strategy(workflow: TradingWorkflow, strategy_id: str) -> Dict[str, Any]:
    """
    Generate a new trading strategy evaluation using the optimized workflow.
    
    Args:
        workflow (TradingWorkflow): The optimized Trading workflow.
        strategy_id (str): The identifier of the trading strategy.
    
    Returns:
        Dict[str, Any]: The generated trading strategy evaluation.
    """
    try:
        evaluation, _ = await workflow.execute_trading_strategy_workflow(strategy_id)
        return evaluation
    except Exception as e:
        logger.error(f"Error generating new strategy for ID '{strategy_id}': {str(e)}")
        return {}

async def apply_trading_strategy(strategy_path: str, strategies: List[str] = None, generate_new: bool = False):
    """
    Apply the optimized trading strategy to review or generate evaluations.
    
    Args:
        strategy_path (str): Path to the JSON file containing the trading strategies.
        strategies (List[str], optional): List of specific strategies to process. If None, process all in the strategy.
        generate_new (bool): If True, generate new evaluations using the optimized workflow.
    """
    strategies_data = load_strategy(strategy_path)
    
    if generate_new:
        # Initialize the workflow for generating new evaluations
        llm_config = {
            "model": "gpt-4-turbo",
            "api_key": "your-api-key-here"  # Replace with actual API key or load from config
        }
        workflow = TradingWorkflow("Trading_Strategy_Optimization", llm_config, "TradingWorkflow")
    
    processed_strategies = set()
    
    for item in strategies_data:
        strategy_id = item.get("strategy_id")
        if not strategy_id:
            logger.warning("Strategy item missing 'strategy_id'. Skipping.")
            continue

        if strategies and strategy_id not in strategies:
            continue
        
        processed_strategies.add(strategy_id)
        
        if generate_new:
            evaluation = await generate_new_strategy(workflow, strategy_id)
            if evaluation:
                print_strategy(strategy_id, evaluation)
            else:
                print(f"Failed to generate new evaluation for: {strategy_id}")
        else:
            evaluation = item.get("evaluation", {})
            if evaluation:
                print_strategy(strategy_id, evaluation)
            else:
                print(f"No evaluation found for: {strategy_id}")
    
    # Process any remaining strategies not found in the strategy file
    if strategies:
        remaining_strategies = set(strategies) - processed_strategies
        if remaining_strategies and generate_new:
            for strategy_id in remaining_strategies:
                evaluation = await generate_new_strategy(workflow, strategy_id)
                if evaluation:
                    print_strategy(strategy_id, evaluation)
                else:
                    print(f"Failed to generate new evaluation for: {strategy_id}")
        elif remaining_strategies:
            logger.warning(f"The following strategies were not found in the strategy file: {remaining_strategies}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply and review optimized trading strategies")
    parser.add_argument("strategy_path", help="Path to the JSON file containing the trading strategies")
    parser.add_argument("--strategies", nargs="*", help="Specific strategies to process")
    parser.add_argument("--generate", action="store_true", help="Generate new evaluations using the optimized workflow")
    args = parser.parse_args()

    import asyncio
    asyncio.run(apply_trading_strategy(args.strategy_path, args.strategies, args.generate))