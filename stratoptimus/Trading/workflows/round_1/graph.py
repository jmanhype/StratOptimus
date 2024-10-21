# -*- coding: utf-8 -*-
# @Date    : 10/04/2024 10:00 AM
# @Author  : issac
# @Desc    : Workflow Classes for Trading Strategy Optimization

from typing import Literal, Dict, Any, Tuple, List
from pydantic import BaseModel, Field
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager
import sys
import os
import asyncio
import json
import vectorbtpro as vbt

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)

import scripts.optimized.Trading.workflows.template.operator as operator
import scripts.optimized.Trading.workflows.round_1.prompt as prompt_custom
from scripts.optimized.Trading.workflows.template.operator import (
    get_parameter_optimizer,
    get_strategy_evaluator
)

# Import the logging module and configure logging
import logging
import traceback

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("workflow_errors.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define Pydantic models for operator responses
class StrategyParameters(BaseModel):
    take_profit: float = Field(default=0.08, description="Take profit percentage")
    stop_loss: float = Field(default=0.12, description="Stop loss percentage")
    sl_window: int = Field(default=400, description="Stop loss window")
    max_orders: int = Field(default=3, description="Maximum number of concurrent orders")
    order_size: float = Field(default=0.0025, description="Order size as a ratio of SOL pool")
    post_buy_delay: int = Field(default=2, description="Delay after a buy order")
    post_sell_delay: int = Field(default=5, description="Delay after a sell order")
    macd_signal_fast: int = Field(default=120, description="MACD fast period")
    macd_signal_slow: int = Field(default=260, description="MACD slow period")
    macd_signal_signal: int = Field(default=90, description="MACD signal period")
    min_macd_signal_threshold: float = Field(default=0.0, description="Minimum MACD signal threshold")
    max_macd_signal_threshold: float = Field(default=0.0, description="Maximum MACD signal threshold")
    enable_sl_mod: bool = Field(default=False, description="Enable stop loss modification")
    enable_tp_mod: bool = Field(default=False, description="Enable take profit modification")

class BacktestResult(BaseModel):
    total_return: float = Field(default=0.0, description="Total return of the strategy")
    sharpe_ratio: float = Field(default=0.0, description="Sharpe ratio of the strategy")
    max_drawdown: float = Field(default=0.0, description="Maximum drawdown of the strategy")

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP", "Trading"]

class Workflow:
    def __init__(
        self,
        name: str,
        llm: Any,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = llm
        self.llm.cost_manager = CostManager()
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str) -> Tuple[str, float]:
        """
        Implementation of the basic workflow

        Args:
            problem (str): The input problem to solve.

        Returns:
            Tuple[str, float]: The solution and the total cost.
        """
        solution = await self.custom(input=problem, instruction="")
        return solution['response'], self.llm.cost_manager.total_cost

class TradingWorkflow(Workflow):
    def __init__(
        self,
        name: str,
        llm: Any,
        dataset: DatasetType,
        trading_data: vbt.Portfolio
    ) -> None:
        super().__init__(name, llm, dataset)
        self.trading_data = trading_data
        self.parameter_optimizer = get_parameter_optimizer(self.llm)
        self.strategy_evaluator = get_strategy_evaluator(self.llm)

    async def execute_trading_workflow(
        self, initial_parameters: StrategyParameters
    ) -> Tuple[Dict[str, Any], float]:
        """
        Executes the trading strategy optimization workflow.

        Args:
            initial_parameters (StrategyParameters): The initial parameters for the trading strategy.

        Returns:
            Tuple[Dict[str, Any], float]: Evaluation results and total cost.
        """
        try:
            # Step 1: Optimize parameters
            optimized_params_response = await self.parameter_optimizer(
                initial_parameters=initial_parameters.dict(),
                trading_data=self.trading_data,
                instruction=prompt_custom.PARAMETER_OPTIMIZATION_PROMPT
            )
            logger.debug(f"Optimized parameters response: {optimized_params_response}")
            optimized_params = StrategyParameters(**self._ensure_dict(optimized_params_response))
            logger.debug(f"Parsed optimized parameters: {optimized_params}")

            # Step 2: Evaluate strategy
            backtest_result_response = await self.strategy_evaluator(
                parameters=optimized_params.dict(),
                trading_data=self.trading_data,
                instruction=prompt_custom.STRATEGY_EVALUATION_PROMPT
            )
            logger.debug(f"Backtest result response: {backtest_result_response}")
            backtest_result = BacktestResult(**self._ensure_dict(backtest_result_response))
            logger.debug(f"Parsed backtest result: {backtest_result}")

            evaluation_results = {
                "optimized_parameters": optimized_params.dict(),
                "backtest_results": backtest_result.dict()
            }

            logger.debug(f"Evaluation results: {evaluation_results}")

            return evaluation_results, self.llm.cost_manager.total_cost
        except Exception as e:
            logger.error(f"Error in execute_trading_workflow: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return {"error": str(e)}, 0.0

    async def __call__(self, initial_parameters: StrategyParameters) -> Tuple[Dict[str, Any], float]:
        """
        Overrides the basic workflow call method to execute the trading workflow.

        Args:
            initial_parameters (StrategyParameters): The initial parameters for the trading strategy.

        Returns:
            Tuple[Dict[str, Any], float]: Evaluation results and total cost.
        """
        return await self.execute_trading_workflow(initial_parameters)

    def _ensure_dict(self, obj: Any) -> Dict[str, Any]:
        """
        Ensures that the provided object is a dictionary.
        If it's a Pydantic model, convert it to a dict.
        If it's a list, convert each element.
        Otherwise, convert the object to a string.

        Args:
            obj (Any): The object to ensure is a dictionary.

        Returns:
            Dict[str, Any]: The standardized dictionary.
        """
        if isinstance(obj, dict):
            return {k: self._ensure_serializable(v) for k, v in obj.items()}
        elif hasattr(obj, 'dict'):
            return {k: self._ensure_serializable(v) for k, v in obj.dict().items()}
        elif isinstance(obj, list):
            return {f"item_{i}": self._ensure_serializable(item) for i, item in enumerate(obj)}
        else:
            return {"value": str(obj)}

    def _ensure_serializable(self, obj: Any) -> Any:
        """
        Recursively ensures that an object is JSON serializable.
        Converts non-serializable objects to strings.

        Args:
            obj (Any): The object to serialize.

        Returns:
            Any: The JSON serializable object.
        """
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return self._ensure_dict(obj)
        elif isinstance(obj, list):
            return [self._ensure_serializable(item) for item in obj]
        else:
            return str(obj)

# Example usage
if __name__ == "__main__":
    llm_config = {
        "model": "gpt-4",
        "api_key": "your-api-key-here"
    }

    # Initialize LLM instance
    llm = create_llm_instance(llm_config)

    # Load trading data (ensure the path is correct)
    trading_data = vbt.Portfolio.from_pickle("path/to/your/trading_data.pickle")

    # Initialize TradingWorkflow
    trading_workflow = TradingWorkflow(
        name="TradingStrategyOptimization",
        llm=llm,
        dataset="Trading",
        trading_data=trading_data
    )

    # Define initial strategy parameters
    initial_params = StrategyParameters(
        take_profit=0.05,
        stop_loss=0.03,
        sl_window=400,
        max_orders=3,
        post_buy_delay=2,
        post_sell_delay=5,
        macd_signal_fast=120,
        macd_signal_slow=260,
        macd_signal_signal=90,
        min_macd_signal_threshold=0.0,
        max_macd_signal_threshold=0.0,
        enable_sl_mod=False,
        enable_tp_mod=False
    )

    # Execute the workflow asynchronously
    async def execute_workflow():
        results, cost = await trading_workflow(initial_params)
        print(f"Optimization results: {results}")
        print(f"Total cost: {cost}")

    asyncio.run(execute_workflow())
