# StratOptimus

**An AI-powered trading strategy optimization framework leveraging recursive self-improvement.**

## Overview

StratOptimus is an advanced trading strategy development and optimization platform that combines machine learning, automated backtesting, and recursive workflow optimization. Built on top of the MetaGPT framework and vectorbtpro, it enables systematic development and refinement of trading strategies through AI-driven iterative improvement.

## Key Features

- **🤖 AI-Driven Strategy Optimization**: Leverages MetaGPT's workflow automation to recursively optimize trading strategies
- **📊 Automated Backtesting**: Built-in backtesting engine using vectorbtpro for strategy validation
- **🔄 Recursive Self-Improvement**: Automatically evolves strategies based on performance feedback
- **⚡ High-Performance Computing**: Designed with JAX integration for efficient numerical computations
- **📈 Performance Analytics**: Comprehensive metrics tracking including Sharpe ratio, returns, and drawdown
- **🔧 Modular Architecture**: Extensible design with pluggable actions, workflows, and operators

## Project Structure

```
StratOptimus/
├── stratoptimus/
│   ├── Trading/                 # Trading-specific modules
│   │   ├── workflows/          # Workflow definitions for strategy optimization
│   │   ├── backtester_module.py
│   │   ├── recursive_optimization.py
│   │   └── apply_strategy.py
│   ├── actions/                # MetaGPT action definitions
│   ├── optimizer_utils/        # Core optimization utilities
│   │   ├── data_utils.py      # Data loading and management
│   │   ├── graph_utils.py     # Workflow graph utilities
│   │   ├── evaluation_utils.py
│   │   ├── experience_utils.py
│   │   └── convergence_utils.py
│   ├── prompts/               # AI prompt templates
│   ├── trading.py             # Core trading evaluation functions
│   └── config_utils.py        # Configuration management
├── .github/
│   └── workflows/             # CI/CD workflows
└── docs/                      # Documentation
```

## Installation

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/StratOptimus.git
   cd StratOptimus
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Key dependencies:
   - `metagpt` - Workflow automation framework
   - `vectorbtpro` - Backtesting engine
   - `jax` - High-performance numerical computing
   - `pandas` - Data manipulation
   - `numpy` - Numerical operations
   - `pyyaml` - Configuration management

3. **Configure the project**

   Create a YAML configuration file with your settings:
   ```yaml
   # config.yaml
   data_path: "./data/trading_data.pkl"
   optimization:
     max_rounds: 10
     convergence_threshold: 0.001
   ```

## Usage

### Basic Example

```python
import asyncio
from stratoptimus.trading import load_data, optimize_trading_evaluation
from stratoptimus.config_utils import load_config

# Load configuration
config = load_config("config.yaml")

# Load trading data
data = asyncio.run(load_data(config['data_path'], samples=1000))

# Run optimization
async def optimize_strategy():
    graph = ...  # Define your trading strategy graph
    avg_score, avg_cost, total_cost = await optimize_trading_evaluation(
        graph=graph,
        file_path=config['data_path'],
        path="./results.csv",
        va_list=data
    )
    print(f"Average Score: {avg_score:.5f}")
    print(f"Total Cost: {total_cost:.5f}")

asyncio.run(optimize_strategy())
```

### Running Recursive Optimization

```bash
python -m stratoptimus.Trading.recursive_optimization
```

This will:
1. Load initial strategy configuration
2. Run backtests on historical data
3. Analyze performance metrics
4. Generate improved strategy variants
5. Iterate until convergence or max rounds reached

## Core Components

### Trading Module (`trading.py`)

Provides core functionality for:
- **Data Loading**: Async loading of trading data from pickle files
- **Strategy Evaluation**: Backtesting using vectorbtpro
- **Concurrent Execution**: Evaluating multiple strategies in parallel
- **Results Management**: Saving and aggregating performance metrics

### Optimizer Utils

- **`data_utils.py`**: Manages optimization results, round selection, and probability distributions
- **`graph_utils.py`**: Handles workflow graph creation, optimization, and file I/O
- **`evaluation_utils.py`**: Performance evaluation and metrics calculation
- **`experience_utils.py`**: Experience replay and learning from past optimization rounds

### Configuration (`config_utils.py`)

Robust configuration management with:
- YAML file loading and validation
- Type checking and error handling
- Clear error messages for debugging

## Development

### Code Style

This project follows:
- **PEP 8**: Python style guidelines
- **PEP 257**: Docstring conventions
- **Type Hints**: Comprehensive type annotations for better IDE support and validation

### Best Practices

- ✅ Functional programming patterns where appropriate
- ✅ Comprehensive error handling with specific exception types
- ✅ Clear, descriptive variable and function names
- ✅ Async/await for concurrent operations
- ✅ Logging for debugging and monitoring
- ✅ Modular, reusable code components

### Testing

Run tests with:
```bash
pytest tests/
```

(Note: Test suite is under development)

## Performance Optimization

StratOptimus is designed for high performance:

- **JAX Integration**: JIT compilation and automatic differentiation for numerical operations
- **Vectorization**: Batch processing of trading strategies
- **Async I/O**: Non-blocking data loading and concurrent backtesting
- **Smart Caching**: Reuse of compiled functions and loaded data

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code:
- Follows the project's code style
- Includes type hints
- Has appropriate error handling
- Includes docstrings for new functions/classes

## Documentation

For detailed documentation, see:
- [Trading Module Documentation](stratoptimus/README.txt)
- [Development Best Practices](stratoptimus/pages/development-process-best-practices/)
- [Architecture Overview](stratoptimus/pages/project-architecture/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on [MetaGPT](https://github.com/geekan/MetaGPT) framework
- Uses [vectorbtpro](https://vectorbt.pro/) for backtesting
- Leverages [JAX](https://github.com/google/jax) for high-performance computing

## Contact

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check existing documentation
- Review the code examples in the repository

---

**Note**: This is an active research project. Trading involves financial risk. Use this software at your own risk and always validate strategies thoroughly before live deployment.
