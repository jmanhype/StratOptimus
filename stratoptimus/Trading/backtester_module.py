#%%
import pandas as pd
import numpy as np
import vectorbtpro as vbt
import random
from numba import njit
from collections import namedtuple
import pickle
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
import os
import pickle
import logging
from typing import Dict, Any

# Configure module-specific logger
logger = logging.getLogger(__name__)

vbt.settings.plotting.use_resampler = True
vbt.settings.wrapping['freq'] = 's'

global trade_memory
TradeMemory = namedtuple("TradeMemory", ["trade_records", "trade_counts"])

def get_best_params(train_portfolio, rand_test_params):
    portfolio = {}
    for asset, pf in train_portfolio.items():
        df = trade_data_dict[asset].copy()
        combined_metrics = pd.DataFrame({
            'total_return': pf.total_return,
            'total_orders': pf.orders.count(),
            'sortino_ratio': pf.sortino_ratio
        })
        
        combined_metrics['total_return'] = combined_metrics['total_return'] / len(df)
        combined_metrics['score'] = combined_metrics['total_orders'] * combined_metrics['sortino_ratio'] * combined_metrics['total_return']
        negative_returns = combined_metrics['total_return'] < 0
        combined_metrics.loc[negative_returns, 'score'] *= 1 / combined_metrics.loc[negative_returns, 'sortino_ratio']    
        portfolio[asset] = combined_metrics

    portfolio_concat = pd.DataFrame()
    for asset, metrics in portfolio.items():
        portfolio_concat = pd.concat([portfolio_concat, metrics])

    grouped = portfolio_concat.groupby(level=portfolio_concat.index.names)
    result = grouped.agg({
        'total_return': 'sum',
        'total_orders': 'sum',
        'sortino_ratio': 'mean',
        'score': 'mean'
    })
    result.sort_values('score', ascending=False)
    result_reset = result.reset_index()
    best_score_params = result_reset.loc[result_reset['score'].idxmax()].to_dict()

    updated_params = rand_test_params.copy()
    for key, value in best_score_params.items():
        if key in updated_params:
            if isinstance(updated_params[key], vbt.Param):
                updated_params[key] = value
            else:
                pass

    updated_params.pop('score', None)
    updated_params.pop('_random_subset', None)

    # Convert float values ending in .0 to int in updated_params
    for key, value in updated_params.items():
        if isinstance(value, float) and value.is_integer():
            updated_params[key] = int(value)

    print("Updated parameters:")
    print(updated_params)

    
    return updated_params

def calculate_stats(test_portfolio, trade_data_dict):
    stats_list = []
    for asset, pf in test_portfolio.items():
        stats = {
            'asset': asset,
            'total_return': pf.total_return,
            'total_pnl': pf.trades.records.pnl.sum(),
            'avg_pnl_per_trade': pf.trades.records['return'].mean(),
            'total_orders': pf.orders.count(),
            'total_trades': pf.trades.count(),
            'sortino_ratio': pf.sortino_ratio,
        }
        stats_list.append(stats)

    all_stats_df = pd.DataFrame(stats_list)
    all_stats_df.set_index('asset', inplace=True)
    all_stats_df = all_stats_df.sort_values('total_return', ascending=True)

    print("All stats DataFrame:")
    print("Top 10 assets:")
    print(all_stats_df.head(10).to_string())
    print("\nBottom 10 assets:")
    print(all_stats_df.tail(10).to_string())
    # display(all_stats_df)

    df = all_stats_df
    name = "All assets"
    total_return_sum = df['total_return'].sum()
    total_pnl_sum = df['total_pnl'].sum()
    avg_sortino_ratio = df['sortino_ratio'].mean()
    avg_pnl_per_trade = total_return_sum / df['total_orders'].sum()
    total_orders = df['total_orders'].sum()
    print(f"\nStatistics for {name}:")
    print(f"Sum of total returns: {total_return_sum:.6f}")
    print(f"Sum of total pnl: {total_pnl_sum:.6f}")
    print(f"Average Sortino ratio: {avg_sortino_ratio:.6f}")
    print(f"Average PnL per trade: {avg_pnl_per_trade:.6f}")
    print(f"Number of assets: {len(df)}")
    print(f"Total orders: {total_orders}")

    return all_stats_df


def load_trade_data(filename: str = 'big_optimize_1016.pkl') -> Dict[str, Any]:
    """
    Load trading data from a pickle file.

    Args:
        filename (str): Name of the pickle file.

    Returns:
        Dict[str, Any]: Loaded trading data dictionary.
    """
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pickle_path = os.path.join(script_dir, filename)
        logger.debug(f"Attempting to load pickle file from: {pickle_path}")

        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                loaded_trade_data_dict = pickle.load(f)
            logger.info(f"Successfully loaded trade data from {pickle_path}")
            logger.info(f"Number of assets loaded: {len(loaded_trade_data_dict)}")
            return loaded_trade_data_dict
        else:
            logger.error(f"Pickle file not found at {pickle_path}")
            return {}
    except Exception as e:
        logger.exception(f"Error loading trade data from {pickle_path}: {e}")
        return {}


@njit
def post_signal_func_nb(c, trade_memory, fees, params):
    if vbt.pf_nb.order_filled_nb(c):
        exit_trade_records = vbt.pf_nb.get_exit_trade_records_nb(c)
        trade_memory.trade_records[: len(exit_trade_records), c.col] = (
            exit_trade_records
        )
        trade_memory.trade_counts[c.col] = len(exit_trade_records)

    # //region set SL init_price to position averaged entry price
    if vbt.pf_nb.order_increased_position_nb(c):
        if params['enable_sl_mod'][0]:
            c.last_sl_info[c.col]["init_price"] = c.last_pos_info[c.col]["entry_price"]
        if params['enable_tp_mod'][0]:
            c.last_tp_info[c.col]["init_price"] = c.last_pos_info[c.col]["entry_price"]


@njit
def get_buy_price(coin_pool_value, sol_pool_value, amount_in_sol):
    major_quantity = coin_pool_value
    minor_quantity = sol_pool_value
    product = minor_quantity * major_quantity
    major_qty_after_execution = product / (minor_quantity + amount_in_sol)
    quantity = major_quantity - major_qty_after_execution
    price = amount_in_sol / quantity

    return price, quantity


@njit
def get_sell_price(coin_pool_value, sol_pool_value, amount_in_coin):
    major_quantity = coin_pool_value
    minor_quantity = sol_pool_value
    product = minor_quantity * major_quantity
    minor_qty_after_execution = product / (major_quantity + amount_in_coin)
    quantity = minor_quantity - minor_qty_after_execution
    price = quantity / amount_in_coin

    return price, quantity


@njit
def signal_func_nb(c, trade_memory, params, data, fees, size, price, last_buy_quantity,last_buy_index,last_sell_index):
    trade_count = trade_memory.trade_counts[c.col]
    trade_records = trade_memory.trade_records[:trade_count, c.col]

    open_trade_records = trade_records[trade_records["status"] == 0]
    closed_trade_records = trade_records[trade_records["status"] == 1]

    num_open_trades = len(open_trade_records)
    num_closed_trades = len(closed_trade_records)
    current_position_size = np.sum(
        open_trade_records["size"] * open_trade_records["entry_price"]
    )

    fees[c.i] = 0

    long_entry = data['entries'][c.i]
    long_exit = False
    size[c.i] = 0

    # region dynamic profit exit
    if num_open_trades > 0:
        current_price = c.close[c.i, 0]
        entry_price = open_trade_records["entry_price"][0]

        open_trade_return = (current_price/ entry_price) - 1
        long_exit = open_trade_return > data['take_profit'][c.i]

        if current_price < entry_price:

            lookback = params['sl_window'][0]
            
            start_index = max(0, c.i - lookback + 1)
            stop_exit_price = np.mean(c.close[start_index:c.i+1, 0])

            open_trade_return = (stop_exit_price / entry_price) - 1.0
            long_exit = open_trade_return < -data['stop_loss'][c.i]


    # endregion
    if long_exit:
        size[c.i] = open_trade_records["size"][0]

    order_size_ratio = params['order_size'][0] 
    order_size = order_size_ratio * data['sol_pool'][c.i]
    current_order_count = np.round(current_position_size / order_size)

    if long_entry and (current_order_count < params['max_orders'][0]):
        size[c.i] = order_size if order_size > 0.01 else 0.0
        long_entry = order_size > 0.01

    else:
        long_entry = False


    action_price = data['sol_pool'][c.i]/data['coin_pool'][c.i]
    
    if long_exit and last_buy_quantity[0][c.col] != 0:
        action_price, sell_quantity = get_sell_price(
            data['coin_pool'][c.i], data['sol_pool'][c.i], last_buy_quantity[0][c.col]
        )
        last_buy_quantity[0][c.col] = 0

    if long_entry:
        action_price, buy_quantity = get_buy_price(
            data['coin_pool'][c.i], data['sol_pool'][c.i], size[c.i][0]
        )
        last_buy_quantity[0][c.col] += buy_quantity

    price[c.i][c.col] = action_price

    if c.index[c.i] <= last_buy_index[0][c.col] + params['post_buy_delay'][0]:
        return False, False, False, False
    
    if long_entry and c.index[c.i] <= last_sell_index[0][c.col] + params['post_sell_delay'][0]:
        long_entry = False

    if long_entry:
        last_buy_index[0][c.col] = data['slot'][c.i]

    if long_exit:
        last_sell_index[0][c.col] = data['slot'][c.i]

    return long_entry, long_exit, False, False


def init_trade_memory(target_shape):
    global trade_memory
    if trade_memory is None:
        trade_memory = TradeMemory(
            trade_records=np.empty(target_shape, dtype=vbt.pf_enums.trade_dt),
            trade_counts=np.full(target_shape[1], 0),
        )
    return trade_memory

# Set a fixed seed for reproducibility
@vbt.parameterized(merge_func="column_stack")
def from_signals_backtest(trade_data_df: pd.DataFrame, **p):
    trade_data_df = calculate_entries_and_params(trade_data_df, p)
    # Add 'Close' column to fix KeyError
    trade_data_df['Close'] = trade_data_df['dex_price']
    
    params_dtype = [
        ('sl_window', np.int32),
        ('max_orders', np.float64),
        ('order_size', np.float64),
        ('enable_sl_mod', np.bool_),
        ('enable_tp_mod', np.bool_),
        ('post_buy_delay', np.int32),
        ('post_sell_delay', np.int32),
    ]
    params = np.array([
        (p['sl_window'], p['max_orders'], p['order_size'],p['enable_sl_mod'],p['enable_tp_mod'],p['post_buy_delay'],p['post_sell_delay'])
    ], dtype=params_dtype)

    data_dtype = [
        ('entries', bool),
        ('stop_loss', np.float64),
        ('take_profit', np.float64),
        ('sol_pool', np.float64),
        ('coin_pool', np.float64),
        ('slot', np.int32),
    ]
    data = np.empty(len(trade_data_df), dtype=data_dtype)
    data['entries'] = trade_data_df['entries'].to_numpy()
    data['stop_loss'] = trade_data_df['stop_loss'].to_numpy()
    data['take_profit'] = trade_data_df['take_profit'].to_numpy()
    data['sol_pool'] = trade_data_df['sol_pool'].to_numpy()
    data['coin_pool'] = trade_data_df['coin_pool'].to_numpy()
    data['slot'] = trade_data_df.index.to_numpy()


    fees = np.full(len(trade_data_df), np.nan)
    size = np.full(len(trade_data_df), np.nan)
    price = trade_data_df["dex_price"].vbt.wrapper.fill().to_numpy()
    last_buy_quantity = 0
    last_buy_index = 0
    last_sell_index = 0

    pf_fs = vbt.Portfolio.from_signals(
        close=trade_data_df["Close"],  # Changed from "dex_price" to "Close"
        size=size,
        price=price,
        jitted=True,
        signal_func_nb=signal_func_nb,
        signal_args=(
            vbt.RepFunc(init_trade_memory),
            params,
            data,
            vbt.Rep("fees"),
            vbt.Rep("size"),
            vbt.Rep("price"),
            vbt.Rep("last_buy_quantity"),
            vbt.Rep("last_buy_index"),
            vbt.Rep("last_sell_index"),
        ),
        post_signal_func_nb=post_signal_func_nb,
        post_signal_args=(vbt.RepFunc(init_trade_memory), vbt.Rep("fees"), params),
        broadcast_named_args=dict(
            last_buy_quantity=last_buy_quantity,
            last_buy_index=last_buy_index,
            last_sell_index=last_sell_index
        ),
        accumulate=True,
        direction=0,
        init_cash=10,
        leverage=np.inf,
        leverage_mode="lazy",
        size_type="value",
        fees=fees,
        from_ago=0
    )
    return pf_fs
#%%
def calculate_entries_and_params(trade_data_df, p):
    # Set default values for take_profit and stop_loss if not provided
    trade_data_df['take_profit'] = p.get('take_profit', 0.1)
    trade_data_df['stop_loss'] = p.get('stop_loss', 0.1)
    
    # Set default values for MACD parameters if not provided
    fast = p.get('macd_signal_fast', 12)
    slow = p.get('macd_signal_slow', 26)
    signal = p.get('macd_signal_signal', 9)
    
    # Calculate MACD and generate signals
    macd = vbt.MACD.run(trade_data_df['dex_price'], fast_window=fast, slow_window=slow, signal_window=signal)
    macd_signal = macd.macd.vbt.crossed_above(macd.signal)
    
    # Ensure 'entries' column is always created
    trade_data_df['entries'] = (
        (macd.macd > p.get('min_macd_signal_threshold', 0)) 
        & macd_signal
    )
    
    # Generate 'exits' column based on MACD crossing below the signal line
    macd_exit_signal = macd.macd.vbt.crossed_below(macd.signal)
    trade_data_df['exits'] = macd_exit_signal
    
    return trade_data_df

#%%
trade_data_dict = load_trade_data("big_optimize_1016.pkl")

if not trade_data_dict:
    print("No trading data loaded. Exiting.")
    sys.exit(1)

########################## TESTING ##########################
params = {
    "take_profit": 0.08,
    "stop_loss": 0.12,
    "sl_window": 400,
    "max_orders": 3,
    "order_size": 0.0025,
    "post_buy_delay": 2,
    "post_sell_delay": 5,
    "macd_signal_fast": 120,
    "macd_signal_slow": 260,
    "macd_signal_signal": 90,
    "min_macd_signal_threshold": 0,
    "max_macd_signal_threshold": 0,
    "enable_sl_mod": False,
    "enable_tp_mod": False,
}

## Optimize params for a few assets and update the params to be used for the whole dataset
# This doesnt need to be run everytime, can manually update params and run the whole backtest
rand_test_params = {
    **params,
    "take_profit": vbt.Param(np.arange(0.01, 0.15, 0.01)), 
    "stop_loss": vbt.Param(np.arange(0.01, 0.15, 0.01)),
    "macd_signal_fast": vbt.Param(np.arange(100, 10000, 100)),
    "macd_signal_slow": vbt.Param(np.arange(100, 10000, 100)),
    "macd_signal_signal": vbt.Param(np.arange(100, 10000, 100)),
    "_random_subset": 200
}

train_portfolio = {}
test_assets = ['RETARDIO','GIGA']
for asset in test_assets:
    if asset in trade_data_dict:
        trade_data = trade_data_dict[asset]
        trade_memory = None

        df = trade_data.copy()
        #ad hoc triming length
        two_weeks_ago = trade_data['timestamp'].max() - pd.Timedelta(weeks=2)
        trade_data = trade_data[trade_data['timestamp'] >= two_weeks_ago]

        pf = from_signals_backtest(trade_data, **rand_test_params)
        train_portfolio[asset] = pf

if train_portfolio:
    params = get_best_params(train_portfolio, rand_test_params)
else:
    print("No assets found for training. Using default params.")
    params = rand_test_params

#%%

#run tests all assets
test_portfolio = {}
for asset, trade_data in trade_data_dict.items():
    trade_memory = None
    trade_data = trade_data.copy()
    #ad hoc triming length
    two_weeks_ago = trade_data['timestamp'].max() - pd.Timedelta(weeks=2)
    trade_data = trade_data[trade_data['timestamp'] >= two_weeks_ago]

    pf = from_signals_backtest(trade_data, **params)
    test_portfolio[asset] = pf


# Calculate stats
all_stats_df = calculate_stats(test_portfolio, trade_data_dict)
# %%
### Some good methods at your disposal
test_portfolio['RETARDIO'].plot()
test_portfolio['RETARDIO'].stats()
test_portfolio['RETARDIO'].trade_history




