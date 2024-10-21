# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 19:46 PM
# @Author  : didi (updated for Trading Strategy Optimization)
# @Desc    : Prompts for operators, including new prompts for Trading Strategy Optimization

SC_ENSEMBLE_PROMPT = """
Several answers have been generated to a same question. They are as follows:
{solutions}

Identify the concise answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

ANSWER_GENERATION_PROMPT = """
Think step by step and solve the problem.
1. In the "thought" field, explain your thinking process in detail.
2. In the "answer" field, provide the final answer concisely and clearly. The answer should be a direct response to the question, without including explanations or reasoning.
Your task: {input}
"""

# New prompts for Trading Strategy Optimization

PARAMETER_OPTIMIZATION_PROMPT = """
Optimize the following trading strategy parameters based on historical performance data. Provide the optimized values for each parameter to maximize the strategy's effectiveness.

Current Parameters:
{current_parameters}

In your response:
1. Suggest optimized values for each parameter.
2. Explain the rationale behind each suggested optimization.
3. Ensure that the proposed changes aim to improve key performance metrics such as total return, Sharpe ratio, and maximum drawdown.

Provide your response in a structured format with each parameter and its suggested optimized value.
"""

STRATEGY_EVALUATION_PROMPT = """
Evaluate the performance of the trading strategy using the provided parameters. Conduct a thorough backtest using historical data and provide comprehensive performance metrics.

Strategy Parameters:
{strategy_parameters}

Trading Data:
{trading_data}

In your response:
1. Calculate and report key performance metrics including:
   - Total Return
   - Sharpe Ratio
   - Maximum Drawdown
   - Win Rate
   - Profit Factor
2. Analyze the strategy's strengths and weaknesses based on the performance metrics.
3. Provide recommendations for further improvements or adjustments to enhance performance.

Present your evaluation in a clear and concise manner, using tables or bullet points where appropriate.
"""

BACKTEST_RESULT_PROMPT = """
Summarize the results of the trading strategy backtest based on the following performance metrics.

Performance Metrics:
- Total Return: {total_return}
- Sharpe Ratio: {sharpe_ratio}
- Maximum Drawdown: {max_drawdown}
- Win Rate: {win_rate}
- Profit Factor: {profit_factor}

In your response:
1. Provide a brief summary of the strategy's overall performance.
2. Highlight the key strengths observed during the backtest.
3. Identify areas where the strategy underperformed or showed potential for improvement.
4. Suggest actionable insights or modifications to enhance future performance.

Ensure that your summary is objective and supported by the provided metrics.
"""

STRATEGY_ADJUSTMENT_PROMPT = """
Based on the backtest results and performance metrics provided, suggest adjustments to the trading strategy parameters to improve its effectiveness.

Backtest Results:
- Total Return: {total_return}
- Sharpe Ratio: {sharpe_ratio}
- Maximum Drawdown: {max_drawdown}
- Win Rate: {win_rate}
- Profit Factor: {profit_factor}

Current Strategy Parameters:
{current_parameters}

In your response:
1. Recommend specific adjustments to one or more strategy parameters.
2. Explain how each suggested adjustment is expected to impact the performance metrics.
3. Ensure that the proposed changes aim to balance return and risk effectively.

Present your recommendations in a structured format, detailing each parameter change and its expected effect.
"""