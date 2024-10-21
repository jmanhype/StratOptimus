# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 17:36 PM
# @Author  : didi
# @Desc    : Operator demo for Trading Strategy Optimization

import ast
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple, Any

from tenacity import retry, stop_after_attempt, wait_fixed

from .operator_an import *
from .op_prompt import *
from metagpt.actions.action_node import ActionNode, CustomJSONEncoder
from metagpt.llm import LLM
from metagpt.logs import logger
import re
import json


class Operator:
    def __init__(self, name, llm: LLM):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    # Enhanced _ensure_serializable method to handle PydanticUndefinedType
    def _ensure_serializable(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._ensure_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._ensure_serializable(obj.__dict__)
        elif obj.__class__.__name__ == 'PydanticUndefinedType':
            return None  # Convert PydanticUndefinedType to None
        else:
            return str(obj)


class Custom(Operator):
    def __init__(self, llm: LLM, name: str = "Custom"):
        super().__init__(name, llm)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        node = await ActionNode.from_pydantic(GenerateOp).fill(context=prompt, llm=self.llm, mode="single_fill")
        response = node.instruct_content.model_dump()
        return response


class AnswerGenerate(Operator):
    def __init__(self, llm: LLM, name: str = "AnswerGenerate"):
        super().__init__(name, llm)

    async def __call__(self, input: str, mode: str = None) -> Tuple[str, str]:
        prompt = ANSWER_GENERATION_PROMPT.format(input=input)
        fill_kwargs = {"context": prompt, "llm": self.llm}
        node = await ActionNode.from_pydantic(AnswerGenerateOp).fill(**fill_kwargs)
        response = node.instruct_content.model_dump()
        return response


class ScEnsemble(Operator):
    def __init__(self, llm: LLM, name: str = "ScEnsemble"):
        super().__init__(name, llm)

    async def __call__(self, solutions: List[str]):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(solutions=solution_text)
        node = await ActionNode.from_pydantic(ScEnsembleOp).fill(context=prompt, llm=self.llm)
        response = node.instruct_content.model_dump()

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping.get(answer, 0)]}


# New operators for Trading Strategy Optimization

class ParameterOptimizer(Operator):
    def __init__(self, llm: LLM, name: str = "ParameterOptimizer"):
        super().__init__(name, llm)

    async def __call__(self, current_parameters: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        prompt = instruction.format(current_parameters=current_parameters)
        node = await ActionNode.from_pydantic(ParameterOptimizerOp).fill(context=prompt, llm=self.llm)
        response = node.instruct_content.model_dump()
        serialized_response = self._ensure_serializable(response)
        return {"optimized_parameters": serialized_response.get("optimized_parameters", {})}


class StrategyEvaluator(Operator):
    def __init__(self, llm: LLM, name: str = "StrategyEvaluator"):
        super().__init__(name, llm)

    async def __call__(self, strategy_parameters: Dict[str, Any], trading_data: Any, instruction: str) -> Dict[str, Any]:
        prompt = instruction.format(strategy_parameters=strategy_parameters, trading_data=trading_data)
        node = await ActionNode.from_pydantic(StrategyEvaluatorOp).fill(context=prompt, llm=self.llm)
        response = node.instruct_content.model_dump()
        serialized_response = self._ensure_serializable(response)
        return {
            "performance_metrics": serialized_response.get("performance_metrics", {}),
            "analysis": serialized_response.get("analysis", "")
        }


class BacktestResult(Operator):
    def __init__(self, llm: LLM, name: str = "BacktestResult"):
        super().__init__(name, llm)

    async def __call__(self, total_return: float, sharpe_ratio: float, max_drawdown: float, win_rate: float, profit_factor: float, instruction: str) -> Dict[str, Any]:
        prompt = instruction.format(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor
        )
        node = await ActionNode.from_pydantic(BacktestResultOp).fill(context=prompt, llm=self.llm)
        response = node.instruct_content.model_dump()
        serialized_response = self._ensure_serializable(response)
        return {
            "total_return": serialized_response.get("total_return", 0.0),
            "sharpe_ratio": serialized_response.get("sharpe_ratio", 0.0),
            "max_drawdown": serialized_response.get("max_drawdown", 0.0),
            "win_rate": serialized_response.get("win_rate", 0.0),
            "profit_factor": serialized_response.get("profit_factor", 0.0)
        }


class StrategyAdjustment(Operator):
    def __init__(self, llm: LLM, name: str = "StrategyAdjustment"):
        super().__init__(name, llm)

    async def __call__(self, backtest_results: Dict[str, float], current_parameters: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        prompt = instruction.format(
            total_return=backtest_results.get("total_return", 0.0),
            sharpe_ratio=backtest_results.get("sharpe_ratio", 0.0),
            max_drawdown=backtest_results.get("max_drawdown", 0.0),
            win_rate=backtest_results.get("win_rate", 0.0),
            profit_factor=backtest_results.get("profit_factor", 0.0),
            current_parameters=current_parameters
        )
        node = await ActionNode.from_pydantic(StrategyAdjustmentOp).fill(context=prompt, llm=self.llm)
        response = node.instruct_content.model_dump()
        serialized_response = self._ensure_serializable(response)
        return {
            "adjusted_parameters": serialized_response.get("adjusted_parameters", {}),
            "expected_impact": serialized_response.get("expected_impact", {})
        }


class NextStepsPlanner(Operator):
    def __init__(self, llm: LLM, name: str = "NextStepsPlanner"):
        super().__init__(name, llm)

    async def __call__(self, backtest_results: Dict[str, float], proposed_adjustments: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        prompt = instruction.format(
            total_return=backtest_results.get("total_return", 0.0),
            sharpe_ratio=backtest_results.get("sharpe_ratio", 0.0),
            max_drawdown=backtest_results.get("max_drawdown", 0.0),
            win_rate=backtest_results.get("win_rate", 0.0),
            profit_factor=backtest_results.get("profit_factor", 0.0),
            proposed_adjustments=proposed_adjustments
        )
        node = await ActionNode.from_pydantic(NextStepsPlannerOp).fill(context=prompt, llm=self.llm)
        response = node.instruct_content.model_dump()
        serialized_response = self._ensure_serializable(response)
        return {
            "plan": serialized_response.get("plan", []),
            "timelines": serialized_response.get("timelines", {}),
            "additional_analyses": serialized_response.get("additional_analyses", []),
            "recommended_tools": serialized_response.get("recommended_tools", [])
        }


# Factory functions to create operator instances

def get_parameter_optimizer(llm: LLM) -> ParameterOptimizer:
    return ParameterOptimizer(llm)

def get_strategy_evaluator(llm: LLM) -> StrategyEvaluator:
    return StrategyEvaluator(llm)

def get_backtest_result(llm: LLM) -> BacktestResult:
    return BacktestResult(llm)

def get_strategy_adjustment(llm: LLM) -> StrategyAdjustment:
    return StrategyAdjustment(llm)

def get_next_steps_planner(llm: LLM) -> NextStepsPlanner:
    return NextStepsPlanner(llm)


# Note: Removed Historical Operators to focus on Trading Strategy Optimization