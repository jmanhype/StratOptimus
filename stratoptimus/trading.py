import asyncio
import pandas as pd
import vectorbtpro as vbt
from typing import Dict, Any, List, Tuple, Optional

async def load_data(file_path: str, samples: int = 1, test: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Load trading data from a pickle file.

    :param file_path: Path to the pickle file containing trading data.
    :param samples: Number of samples to load.
    :param test: Indicates whether to load test data.
    :return: Dictionary mapping dataset names to their corresponding DataFrames.
    """
    try:
        data = pd.read_pickle(file_path)
        if test:
            # Assuming test data is stored with key 'test'
            loaded_data = {'test': data['test'].head(samples)}
        else:
            # Assuming training data is stored with key 'train'
            loaded_data = {'train': data['train'].head(samples)}
        return loaded_data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The file at {file_path} was not found.") from e
    except KeyError as e:
        raise KeyError(f"Expected key not found in data file {file_path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading data from {file_path}: {e}") from e

async def evaluate_problem(problem: Dict[str, Any], graph: Any) -> Tuple[float, float]:
    """
    Evaluate a single trading strategy using backtesting.

    :param problem: A dictionary containing problem parameters.
    :param graph: The trading strategy graph or configuration.
    :return: A tuple containing the performance metric and cost.
    :raises ValueError: If required data fields are missing from problem dict.
    :raises RuntimeError: If backtest execution fails.
    """
    try:
        # Extract parameters from the problem dictionary
        parameters = problem.get('parameters', {})
        price_data = problem.get('price_data')

        if price_data is None:
            raise ValueError("Missing required 'price_data' field in problem dictionary")

        # Initialize the strategy using vectorbtpro
        strategy = vbt.Strategy.from_graph(graph)

        # Run the backtest
        portfolio = strategy.run(price=price_data, **parameters)

        # Calculate performance metrics
        performance = portfolio.total_return()  # Example metric
        cost = portfolio.total_fees  # Example cost metric

        return float(performance), float(cost)
    except ValueError as e:
        # Re-raise validation errors
        raise
    except Exception as e:
        # Log error and return default metrics for robustness
        print(f"Error evaluating problem: {e}")
        return 0.0, 0.0

async def evaluate_all_problems(
    data: Dict[str, pd.DataFrame],
    graph: Any,
    max_concurrent_tasks: int = 25
) -> List[Tuple[float, float]]:
    """
    Evaluate all trading problems concurrently.

    :param data: Dictionary of datasets to evaluate.
    :param graph: The trading strategy graph or configuration.
    :param max_concurrent_tasks: Maximum number of concurrent evaluation tasks.
    :return: List of tuples containing performance metrics and costs for each problem.
    """
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    results = []

    async def evaluate_with_semaphore(problem_data: Dict[str, Any]):
        async with semaphore:
            result = await evaluate_problem(problem_data, graph)
            results.append(result)

    tasks = []
    for dataset_name, df in data.items():
        for _, row in df.iterrows():
            problem = row.to_dict()
            tasks.append(asyncio.create_task(evaluate_with_semaphore(problem)))

    await asyncio.gather(*tasks)
    return results

def save_results_to_csv(results: List[Tuple[float, float]], path: str) -> Tuple[float, float, float]:
    """
    Save backtesting results to a CSV file and calculate average metrics.

    :param results: List of tuples containing performance metrics and costs.
    :param path: Path to save the CSV file.
    :return: Tuple containing average score, average cost, and total cost.
    """
    df = pd.DataFrame(results, columns=['Performance', 'Cost'])
    df.to_csv(path, index=False)

    average_score = df['Performance'].mean()
    average_cost = df['Cost'].mean()
    total_cost = df['Cost'].sum()

    return average_score, average_cost, total_cost

async def optimize_trading_evaluation(
    graph: Any,
    file_path: str,
    path: str,
    va_list: List[Any]
) -> Tuple[float, float, float]:
    """
    Optimize trading strategy evaluation by running backtests and saving results.

    :param graph: The trading strategy graph or configuration.
    :param file_path: Path to the trading data file.
    :param path: Path to save the backtest results.
    :param va_list: List of variables/parameters for backtesting.
    :return: Tuple containing average score, average cost, and total cost.
    """
    # Load trading data
    data = await load_data(file_path, samples=len(va_list))
    
    # Evaluate all problems concurrently
    results = await evaluate_all_problems(data, graph, max_concurrent_tasks=25)
    
    # Save results to CSV and calculate average metrics
    average_score, average_cost, total_cost = save_results_to_csv(results, path=path)
    
    print(f"Average score on Trading dataset: {average_score:.5f}")
    print(f"Total Cost: {total_cost:.5f}")
    
    return average_score, average_cost, total_cost
