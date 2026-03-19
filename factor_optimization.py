"""
因子优化与合成回测框架 - 工具函数模块
"""
import json
import pandas as pd
import numpy as np
from factor_analysis import (
    winsorize, standardize, run_backtest, compute_performance,calc_jump_volatility_ratio,
    calc_jump_momentum, calc_jump_skewness, calc_jump_bipower
)

# ==================== 因子函数映射字典 ====================
FACTOR_FUNCTIONS = {
    "jump_bipower": calc_jump_bipower,
    "jump_volatility_ratio": calc_jump_volatility_ratio,
    "jump_momentum": calc_jump_momentum,
    "jump_skewness": calc_jump_skewness,

}

# ==================== 回测参数 ====================
REBALANCE_FREQ = 'M'          # 调仓频率
BT_START = "2015-01-01"       # 回测开始日期
BT_END = "2021-12-31"         # 回测结束日期
INIT_CAPITAL = 1e6

def compute_factor(close, factor_func, params):
    """
    根据因子函数和参数字典计算因子值，并进行去极值、标准化处理
    """
    factor_raw = factor_func(close, **params)
    factor_raw = factor_raw.dropna(axis=1, how='all')
    factor_proc = winsorize(factor_raw)
    factor_proc = standardize(factor_proc)
    return factor_proc

def backtest_factor(factor, forward_ret, name=""):
    """
    对单个因子进行回测，返回业绩指标字典
    """
    common_idx = factor.index.intersection(forward_ret.index)
    if len(common_idx) == 0:
        print(f"⚠️ {name}: 日期无重叠，跳过")
        return None
    f = factor.loc[common_idx]
    fr = forward_ret.loc[common_idx]

    try:
        nav, _ = run_backtest(f, fr, rebalance_freq=REBALANCE_FREQ,
                               start_date=BT_START, end_date=BT_END,
                               initial_capital=INIT_CAPITAL)
        nav = pd.to_numeric(nav, errors='coerce').dropna()
        if len(nav) == 0:
            print(f"⚠️ {name}: 回测无有效净值")
            return None
        perf = compute_performance(nav)
        return perf
    except Exception as e:
        print(f"⚠️ {name}: 回测出错 - {e}")
        return None

# ==================== 参数配置 ====================
TARGET_FACTOR_NAME = "jump_volatility_ratio"       # 目标因子名称（可从字典的键中选择）
DEFAULT_PARAMS = {"short_window":5, "long_window":25}  # 默认参数（仅用于对比）

PARAM_GRID = {
    "short_window": [i for i in range(5)],
    "long_window": [i for i in range(20,40)],

}

OUTPUT_BEST_PARAMS_FILE = "best_params.json"   # 最佳参数保存文件

# ==================== 主流程 ====================
if __name__ == "__main__":
    # 1. 读取数据
    print("读取收盘价宽表...")
    close_wide = pd.read_parquet("zz500_close_wide.parquet", engine='fastparquet')
    close_wide.index = pd.to_datetime(close_wide.index)
    close_wide.sort_index(inplace=True)

    forward_ret = close_wide.pct_change().shift(-1)

    # 2. 从字典中获取目标因子函数
    target_func = FACTOR_FUNCTIONS.get(TARGET_FACTOR_NAME)
    if target_func is None:
        raise ValueError(f"未找到因子函数 {TARGET_FACTOR_NAME}，请检查 FACTOR_FUNCTIONS 字典")

    print("\n========== 参数网格搜索 ==========")
    best_sharpe = -np.inf
    best_params = None
    best_perf = None
    results = []

    for sw in PARAM_GRID["short_window"]:
        for lw in PARAM_GRID["long_window"]:
            params = {"short_window":sw, "long_window":lw}
            factor = compute_factor(close_wide, target_func, params)
            perf = backtest_factor(factor, forward_ret, name=f"{TARGET_FACTOR_NAME}(sw={sw},lw={lw})")
            if perf is not None:
                sharpe = perf["夏普比率"]
                results.append({
                    "short_window": sw,
                    "long_window": lw,
                    "年化收益率": perf["年化收益率"],
                    "夏普比率": sharpe,
                    "最大回撤": perf["最大回撤"]
                })
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
                    best_perf = perf

    # 输出结果
    if results:
        df_results = pd.DataFrame(results)
        print("\n参数网格搜索业绩汇总：")
        print(df_results.round(4))

        print(f"\n最佳参数组合: {best_params}")
        print(f"最佳夏普比率: {best_sharpe:.4f}")
        print(f"对应年化收益: {best_perf['年化收益率']:.2%}, 最大回撤: {best_perf['最大回撤']:.2%}")

        # 保存最佳参数到文件
        with open(OUTPUT_BEST_PARAMS_FILE, "w") as f:
            json.dump(best_params, f)
        print(f"最佳参数已保存至 {OUTPUT_BEST_PARAMS_FILE}")
    else:
        print("参数搜索未产生有效结果")