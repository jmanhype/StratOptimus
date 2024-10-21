# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 19:46 PM
# @Author  : didi (updated for Trading Strategy Optimization)
# @Desc    : Action nodes for operators, including new nodes for Trading Strategy Optimization

from pydantic import BaseModel, Field
from typing import List, Dict, Any

class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")

class ScEnsembleOp(BaseModel):
    thought: str = Field(default="", description="The thought of the most consistent solution.")
    solution_letter: str = Field(default="", description="The letter of the most consistent solution.")

class AnswerGenerateOp(BaseModel):
    thought: str = Field(default="", description="The step-by-step thinking process")
    answer: str = Field(default="", description="The final answer to the question")

# New action nodes for Trading Strategy Optimization

class ParameterOptimizerOp(BaseModel):
    optimized_parameters: Dict[str, Any] = Field(default_factory=dict, description="Optimized trading strategy parameters")

class StrategyEvaluatorOp(BaseModel):
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics of the trading strategy")
    analysis: str = Field(default="", description="Analysis of the strategy's strengths and weaknesses")

class BacktestResultOp(BaseModel):
    total_return: float = Field(default=0.0, description="Total return of the trading strategy")
    sharpe_ratio: float = Field(default=0.0, description="Sharpe ratio of the trading strategy")
    max_drawdown: float = Field(default=0.0, description="Maximum drawdown of the trading strategy")
    win_rate: float = Field(default=0.0, description="Win rate of the trading strategy")
    profit_factor: float = Field(default=0.0, description="Profit factor of the trading strategy")

class StrategyAdjustmentOp(BaseModel):
    adjusted_parameters: Dict[str, Any] = Field(default_factory=dict, description="Adjusted trading strategy parameters")
    expected_impact: Dict[str, Any] = Field(default_factory=dict, description="Expected impact of the adjustments on performance metrics")

class NextStepsPlannerOp(BaseModel):
    plan: List[str] = Field(default_factory=list, description="Step-by-step plan for further optimization")
    timelines: Dict[str, str] = Field(default_factory=dict, description="Timelines and milestones for each step")
    additional_analyses: List[str] = Field(default_factory=list, description="Additional analyses or tests to conduct")
    recommended_tools: List[str] = Field(default_factory=list, description="Recommended tools or resources for optimization")
