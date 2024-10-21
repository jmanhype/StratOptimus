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

PLANNING_NEXT_STEPS_PROMPT = """
Plan the next steps for further optimizing the trading strategy based on the recent backtest results and proposed parameter adjustments.

Recent Backtest Results:
- Total Return: {total_return}
- Sharpe Ratio: {sharpe_ratio}
- Maximum Drawdown: {max_drawdown}
- Win Rate: {win_rate}
- Profit Factor: {profit_factor}

Proposed Parameter Adjustments:
{proposed_adjustments}

In your response:
1. Outline a step-by-step plan to implement the proposed parameter adjustments.
2. Include timelines and milestones for each step.
3. Suggest any additional analyses or tests that should be conducted to validate the effectiveness of the adjustments.
4. Recommend tools or resources that may aid in the optimization process.

Ensure that the plan is actionable, detailed, and aimed at systematically enhancing the trading strategy's performance.
"""

WORKFLOW_OPTIMIZE_PROMPT = """You are building a Graph and corresponding Prompt to jointly solve trading strategy optimization problems. 
Referring to the given graph and prompt, which forms a basic example of a trading strategy optimization approach, 
please reconstruct and optimize them. You can add, modify, or delete nodes, parameters, or prompts. Include your 
single modification in XML tags in your reply. Ensure they are complete and correct to avoid runtime failures. When 
optimizing, you can incorporate critical thinking methods like review, revise, ensemble (generating multiple answers through different/similar prompts, then voting/integrating/checking the majority to obtain a final answer), selfAsk, etc. Consider 
Python's loops (for, while, list comprehensions), conditional statements (if-elif-else, ternary operators), 
or machine learning techniques (e.g., linear regression, decision trees, neural networks, clustering). The graph 
complexity should not exceed 10. Use logical and control flow (IF-ELSE, loops) for a more enhanced graphical 
representation. Ensure that all the prompts required by the current graph from prompt_custom are included. Exclude any other prompts.
Output the modified graph and all the necessary Prompts in prompt_custom (if needed).
The prompt you need to generate is only the one used in `prompt_custom.XXX` within Custom. Other methods already have built-in prompts and are prohibited from being generated. Only generate those needed for use in `prompt_custom`; please remove any unused prompts in prompt_custom.
The generated prompt must not contain any placeholders.
Considering information loss, complex graphs may yield better results, but insufficient information transmission can omit the solution. It's crucial to include necessary context during the process."""

WORKFLOW_INPUT = """
Here is a graph and the corresponding prompt (prompt only related to the custom method) that performed excellently in a previous iteration (maximum score is 1). You must make further optimizations and improvements based on this graph. The modified graph must differ from the provided example, and the specific differences should be noted within the <modification>xxx</modification> section.

<sample>
    <experience>{experience}</experience>
    <modification>(such as: add a review step/delete a operator/modify a prompt)</modification>
    <score>{score}</score>
    <graph>{graph}</graph>
    <prompt>{prompt}</prompt> <!-- only prompt_custom -->
    <operator_description>{operator_description}</operator_description>
</sample>

Below are the logs of some results with the aforementioned Graph that performed well but encountered errors, which can be used as references for optimization:
{log}

First, provide optimization ideas. **Only one detail point can be modified at a time**, and no more than 5 lines of code may be changed per modification—extensive modifications are strictly prohibited to maintain project focus!
When introducing new functionalities in the graph, please make sure to import the necessary libraries or modules yourself, except for operator, prompt_custom, create_llm_instance, and CostManager, which have already been automatically imported.
**Under no circumstances should Graph output None for any field.**
Use custom methods to restrict your output format, rather than using code (outside of the code, the system will extract answers based on certain rules and score them).
It is very important to format the Graph output answers, you can refer to the standard answer format in the log.
"""

WORKFLOW_CUSTOM_USE = """\nHere's an example of using the `custom` method in graph:
```
# You can write your own prompt in <prompt>prompt_custom</prompt> and then use it in the Custom method in the graph
response = await self.custom(input=problem, instruction=prompt_custom.PARAMETER_OPTIMIZATION_PROMPT)
# You can also concatenate previously generated string results in the input to provide more comprehensive contextual information.
# response = await self.custom(input=problem+f"xxx:{xxx}, xxx:{xxx}", instruction=prompt_custom.PARAMETER_OPTIMIZATION_PROMPT)
# The output from the Custom method can be placed anywhere you need it, as shown in the example below
solution = await self.generate(problem=f"question:{problem}, optimized_params:{response['response']}")
```
Note: In custom, the input and instruction are directly concatenated (instruction + input), and placeholders are not supported. Please ensure to add comments and handle the concatenation externally.

**Introducing multiple operators at appropriate points can enhance performance. If you find that some provided operators are not yet used in the graph, try incorporating them.**
"""

WORKFLOW_TEMPLATE = """from typing import Literal
import scripts.optimized.Trading.workflows.template.operator as operator
import scripts.optimized.Trading.workflows.round_{round}.prompt as prompt_custom
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP", "Trading"]

{graph}
"""