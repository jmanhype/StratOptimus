# -*- coding: utf-8 -*-
# @Date    : 10/19/2024
# @Author  : Your Name
# @Desc    : Recursive Optimization for Trading Strategy Optimization

import asyncio
import json
from typing import Dict, Any, Tuple, List
import pandas as pd
import logging
import vectorbtpro as vbt
from scripts.optimized.Trading.workflows.round_1.graph import StrategyParameters
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager
import concurrent.futures
from scripts.optimized.Trading.backtester_module import from_signals_backtest, load_trade_data, calculate_entries_and_params  # Correct import path
import numpy as np  # Ensure numpy is imported for performance evaluation
from langsmith import traceable
from langsmith.wrappers import wrap_openai
import openai
from scripts.config_utils import load_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clip_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clip parameters to ensure they stay within predefined bounds.
    
    Args:
        params (Dict[str, Any]): Parameters to be clipped.
    
    Returns:
        Dict[str, Any]: Clipped parameters.
    """
    clipped = params.copy()
    bounds = {
        "ma_window": (5, 200),
        "fees": (0.0001, 0.01),
        "take_profit": (0.01, 0.5),
        "stop_loss": (0.01, 0.5),
        "sl_window": (100, 1000),
        "max_orders": (1, 10),
        "post_buy_delay": (0, 10),
        "post_sell_delay": (0, 10),
        "macd_signal_fast": (10, 300),
        "macd_signal_slow": (10, 300),
        "macd_signal_signal": (10, 300),
        "min_macd_signal_threshold": (0.0, 1.0),
        "max_macd_signal_threshold": (0.0, 1.0),
    }
    
    for param, (min_val, max_val) in bounds.items():
        if param in clipped:
            try:
                clipped[param] = max(min_val, min(float(clipped[param]), max_val))
            except ValueError:
                logger.warning(f"Invalid value for parameter '{param}': {clipped[param]}. Using default bounds.")
                clipped[param] = max(min_val, min(min_val, max_val))
    
    return clipped

async def adjust_parameters(current_params: Dict[str, Any], performance: float) -> Dict[str, Any]:
    """
    Adjust parameters based on current performance for the trading strategy.
    
    Args:
        current_params (Dict[str, Any]): Current parameters.
        performance (float): Current performance score (average total return).
    
    Returns:
        Dict[str, Any]: Adjusted parameters.
    """
    adjusted_params = current_params.copy()
    
    # Adjust moving average window
    if performance < 0:
        # If performance is negative, try a shorter MA window
        adjusted_params['ma_window'] = max(5, current_params['ma_window'] - 5)
    elif performance < 0.05:
        # If performance is low but positive, try a longer MA window
        adjusted_params['ma_window'] = min(200, current_params['ma_window'] + 5)
    
    # Adjust fees
    if performance < 0.01:
        # If performance is very low, slightly reduce fees to see if it helps
        adjusted_params['fees'] = max(0.0001, current_params['fees'] * 0.95)
    
    # Add more parameter adjustments here as needed
    
    return adjusted_params

def parse_llm_response(response: str, current_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the LLM response to update parameters.
    Handles both JSON and free-form text responses by enforcing JSON formatting.
    
    Args:
        response (str): LLM response containing parameter suggestions.
        current_params (Dict[str, Any]): Current parameters.
    
    Returns:
        Dict[str, Any]: Updated parameters.
    """
    updated_params = current_params.copy()
    try:
        # Try to parse the entire response as JSON
        suggested_updates = json.loads(response)
        for key, value in suggested_updates.items():
            if key in current_params:
                updated_params[key] = float(value)
                logger.info(f"Parameter '{key}' updated to {value}")
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract key-value pairs manually
        logger.warning("Failed to parse LLM response as JSON. Attempting to extract parameters manually.")
        lines = response.split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().strip('"').strip("'")
                value = value.strip().strip(',').strip('"').strip("'")
                if key in current_params:
                    try:
                        updated_value = float(value)
                        updated_params[key] = updated_value
                        logger.info(f"Parameter '{key}' updated to {updated_value}")
                    except ValueError:
                        logger.warning(f"Invalid value for parameter '{key}': {value}. Skipping update.")
    
    # Clip parameters to stay within bounds
    updated_params = clip_parameters(updated_params)
    
    return updated_params

def evaluate_performance(portfolios: Dict[str, vbt.Portfolio]) -> float:
    """
    Evaluate the performance of multiple portfolios.
    
    Args:
        portfolios (Dict[str, vbt.Portfolio]): Dictionary mapping asset symbols to their portfolios.
    
    Returns:
        float: Average total return across all portfolios.
    """
    total_returns = [pf.total_return for pf in portfolios.values()]
    return np.mean(total_returns)

@traceable
async def recursive_optimization(
    initial_params: Dict[str, Any],
    dataset: Any,
    llm: Any,
    trading_data: Dict[str, pd.DataFrame],
    max_iterations: int = 45,
    convergence_threshold: float = 0.01,
    parallel_evaluations: int = 5
) -> Tuple[Dict[str, Any], float]:
    """
    Perform recursive optimization to improve trading strategies.
    
    Args:
        initial_params (Dict[str, Any]): Initial parameters for strategy generation.
        dataset (Any): Loaded trading dataset (list of asset symbols).
        llm (Any): Language model instance.
        trading_data (Dict[str, pd.DataFrame]): Trading data for backtesting.
        max_iterations (int): Maximum number of optimization iterations.
        convergence_threshold (float): Threshold for early stopping if improvement is minimal.
        parallel_evaluations (int): Number of parallel strategy evaluations.
    
    Returns:
        Tuple[Dict[str, Any], float]: Best strategy and its performance score.
    """
    best_strategy = initial_params.copy()
    best_performance = float('-inf')
    current_params = initial_params.copy()
    performance_history = []

    def evaluate_strategy(params):
        portfolios = apply_trading_strategy(trading_data, params)
        performance = evaluate_performance(portfolios)
        return params, performance

    for iteration in range(1, max_iterations + 1):
        logger.info(f"Starting iteration {iteration}/{max_iterations}")
        
        # Generate multiple parameter sets
        parameter_sets = [current_params.copy() for _ in range(parallel_evaluations)]
        for idx, params in enumerate(parameter_sets[1:], start=2):
            adjusted = await adjust_parameters(params, best_performance)
            logger.debug(f"Adjusted parameters for parallel evaluation {idx}: {adjusted}")

        # Evaluate strategies in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_evaluations) as executor:
            results = list(executor.map(evaluate_strategy, parameter_sets))

        # Process results
        for params, performance in results:
            logger.info(f"Performance for parameters {params}: {performance}")
            if performance > best_performance:
                best_performance = performance
                best_strategy = params.copy()
                logger.info(f"New best strategy found with performance {best_performance}")
    
        performance_history.append((iteration, best_performance))
        logger.info(f"Iteration {iteration}: Best performance so far: {best_performance}")
        
        # Use LLM to suggest parameter adjustments for the best strategy
        updated_params = await get_llm_suggestions(llm, current_params, best_performance, best_strategy, iteration)
        
        # Check for convergence
        if best_strategy:
            improvements = {k: abs(updated_params[k] - best_strategy[k]) / best_strategy[k] for k in updated_params}
            max_improvement = max(improvements.values(), default=0)
            logger.debug(f"Max relative improvement in this iteration: {max_improvement}")
            if max_improvement < convergence_threshold and iteration > max_iterations // 2:
                logger.info("Convergence reached. Stopping optimization.")
                break
        
        current_params = updated_params

    # Log performance history
    performance_df = pd.DataFrame(performance_history, columns=['Iteration', 'Performance'])
    performance_df.to_csv('optimization_history.csv', index=False)
    logger.info(f"Optimization history saved to 'optimization_history.csv'")

    return best_strategy, best_performance

def from_signals_backtest(trade_data: pd.DataFrame, **params) -> vbt.Portfolio:
    """
    Create a Portfolio from signals using the backtester's from_signals_backtest function.
    
    Args:
        trade_data (pd.DataFrame): Trading data with 'timestamp' and other indicators.
        **params: Strategy parameters.
    
    Returns:
        vbt.Portfolio: Backtested portfolio.
    """
    # Ensure 'dex_price' column exists
    if 'dex_price' not in trade_data.columns:
        raise ValueError(f"'dex_price' column not found. Available columns: {trade_data.columns}")

    # Assuming 'trade_data' has columns 'entries' and 'exits' generated by the backtester
    entries = trade_data['entries']
    exits = trade_data['exits']
    
    portfolio = vbt.Portfolio.from_signals(
        close=trade_data['dex_price'],  # Changed from 'Close' to 'dex_price'
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=params.get('fees', 0.001)
    )
    return portfolio

@traceable
def apply_trading_strategy(trading_data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, vbt.Portfolio]:
    portfolios = {}
    for asset, trade_data in trading_data.items():
        try:
            trade_data = trade_data.copy()
            
            # Calculate entries and exits based on your strategy
            # This is a placeholder - replace with your actual strategy logic
            trade_data['entries'] = (trade_data['dex_price_pct_change'] > params.get('entry_threshold', 0.01))
            trade_data['exits'] = (trade_data['dex_price_pct_change'] < params.get('exit_threshold', -0.01))
            
            two_weeks_ago = trade_data['timestamp'].max() - pd.Timedelta(weeks=2)
            trade_data = trade_data[trade_data['timestamp'] >= two_weeks_ago]
            
            pf = from_signals_backtest(trade_data, **params)
            portfolios[asset] = pf
        except Exception as e:
            logger.error(f"Error processing asset {asset}: {str(e)}")
            continue
    
    return portfolios

@traceable
async def get_llm_suggestions(llm, current_params, best_performance, best_strategy, iteration):
    """
    Traceable function to get LLM suggestions.
    """
    prompt = f"""
    Current parameters: {best_strategy}
    Current performance: {best_performance}
    Previous iterations: {iteration}

    Please suggest new adjustments to improve performance. Aim for diversity in your suggestions.
    Respond ONLY with a JSON object containing parameter names as keys and their new values. Do not include any explanatory text.

    Example response:
    {{
        "ma_window": 35,
        "fees": 0.00045
    }}
    """
    logger.debug(f"Prompt sent to LLM: {prompt}")
    try:
        response = await llm.aask(prompt)
        logger.debug(f"Response received from LLM: {response}")
        
        # Parse LLM response and update parameters
        updated_params = parse_llm_response(response, best_strategy)
        
        logger.info(f"Adjusted parameters: {updated_params}")
        
        return updated_params
    except Exception as e:
        logger.error(f"Error during LLM interaction or parameter update: {e}")
        # Handle exception as needed
        return current_params

@traceable
async def run_optimization():
    # Load configuration
    config = load_config("config/trading_config.yaml")
    
    # Initialize parameters from config
    initial_params = config.get("initial_params", {})
    dataset = config.get("dataset", [])  # Ensure this is defined in your config
    max_iterations = config.get("max_rounds", 45)
    convergence_threshold = config.get("convergence_threshold", 0.01)
    
    # Create LLM instance
    llm_config = config.get("llm_config", {})
    llm = create_llm_instance(llm_config)
    
    # Load trading data
    trading_data = load_trade_data(config.get("trading_data_path", "trading_data/trading_data.pickle"))
    
    # Run recursive optimization
    best_strategy, best_performance = await recursive_optimization(
        initial_params,
        dataset,
        llm,
        trading_data,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold
    )
    
    # Log or process results
    logger.info(f"Best strategy: {best_strategy}")
    logger.info(f"Best performance: {best_performance}")
    
    # Additional execution logic here
    ...

if __name__ == "__main__":
    config = load_config("config/trading_config.yaml")
    openai_client = wrap_openai(openai.Client(api_key=config["llm_config"]["api_key"]))
    asyncio.run(run_optimization())
