"""
因子均值合成优化脚本
- 加载网格搜索得到的最佳参数（如存在）
- 计算默认参数因子、最优参数因子、合成因子
- 输出对比表格
"""
import pandas as pd
import numpy as np
import json
from factor_optimization import (
    compute_factor, backtest_factor, FACTOR_FUNCTIONS,  # 导入因子函数字典
    REBALANCE_FREQ, BT_START, BT_END, INIT_CAPITAL
)

# ==================== 参数配置 ====================
TARGET_FACTOR_NAME = "jump_intensity"
DEFAULT_PARAMS = {"window": 20, "threshold": 2}

# 合成因子配置列表 (名称, 参数字典) - 只定义名称和参数，函数从FACTOR_FUNCTIONS获取
SYNTHESIS_FACTOR_CONFIGS = [
    ("jump_intensity", {"window": 20, "threshold": 2}),
    ("jump_volatility_ratio", {"short_window": 5, "long_window": 20}),
    ("jump_momentum", {"window": 20, "momentum_window": 5}),
    # 可继续添加其他因子配置
]

BEST_PARAMS_FILE = "best_params.json"

# ==================== 主流程 ====================
if __name__ == "__main__":
    # 1. 读取数据
    print("读取收盘价宽表...")
    close_wide = pd.read_parquet("zz500_close_wide.parquet", engine='fastparquet')
    close_wide.index = pd.to_datetime(close_wide.index)
    close_wide.sort_index(inplace=True)

    forward_ret = close_wide.pct_change().shift(-1)

    # 2. 尝试加载最佳参数
    best_params = DEFAULT_PARAMS.copy()
    try:
        with open(BEST_PARAMS_FILE, "r") as f:
            loaded = json.load(f)
            best_params = {k: loaded.get(k, DEFAULT_PARAMS[k]) for k in DEFAULT_PARAMS.keys()}
        print(f"已加载最佳参数: {best_params}")
    except FileNotFoundError:
        print(f"未找到 {BEST_PARAMS_FILE}，将使用默认参数作为最优参数（回退）")
    except Exception as e:
        print(f"加载最佳参数出错: {e}，使用默认参数")

    # 3. 获取目标因子函数
    target_func = FACTOR_FUNCTIONS.get(TARGET_FACTOR_NAME)
    if target_func is None:
        raise ValueError(f"目标因子 {TARGET_FACTOR_NAME} 不可用，请检查 FACTOR_FUNCTIONS")

    # 4. 计算目标因子（默认参数）
    print("\n========== 默认因子 ==========")
    default_factor = compute_factor(close_wide, target_func, DEFAULT_PARAMS)
    default_perf = backtest_factor(default_factor, forward_ret, name=f"{TARGET_FACTOR_NAME}(默认)")
    if default_perf is None:
        default_perf = {"年化收益率": np.nan, "夏普比率": np.nan, "最大回撤": np.nan}

    # 5. 计算目标因子（最优参数）
    print("\n========== 最优参数因子 ==========")
    best_factor = compute_factor(close_wide, target_func, best_params)
    best_perf = backtest_factor(best_factor, forward_ret, name=f"{TARGET_FACTOR_NAME}(最优)")
    if best_perf is None:
        best_perf = {"年化收益率": np.nan, "夏普比率": np.nan, "最大回撤": np.nan}

    # 6. 合成因子回测（仅使用FACTOR_FUNCTIONS中存在的因子）
    print("\n========== 合成因子 ==========")
    component_factors = []
    for name, params in SYNTHESIS_FACTOR_CONFIGS:
        func = FACTOR_FUNCTIONS.get(name)
        if func is None:
            print(f"⚠️ 跳过因子 {name}，因为该因子不可用")
            continue
        print(f"计算成分因子: {name}")
        f = compute_factor(close_wide, func, params)
        component_factors.append(f)

    if len(component_factors) == 0:
        print("⚠️ 没有可用的合成因子，跳过合成回测")
        synthesis_perf = {"年化收益率": np.nan, "夏普比率": np.nan, "最大回撤": np.nan}
    else:
        # 取所有因子日期的交集
        common_dates = component_factors[0].index
        for f in component_factors[1:]:
            common_dates = common_dates.intersection(f.index)

        # 对齐后取均值合成
        aligned_factors = [f.loc[common_dates] for f in component_factors]
        # 取所有因子列的交集
        common_cols = set.intersection(*[set(f.columns) for f in aligned_factors])
        aligned_factors = [f[list(common_cols)] for f in aligned_factors]

        # 等权合成
        synthesis_factor = pd.concat(aligned_factors, axis=0).groupby(level=0).mean()
        synthesis_perf = backtest_factor(synthesis_factor, forward_ret, name="合成因子(等权)")
        if synthesis_perf is None:
            synthesis_perf = {"年化收益率": np.nan, "夏普比率": np.nan, "最大回撤": np.nan}

    # 7. 输出对比表格
    print("\n========== 最终对比 ==========")
    comparison = pd.DataFrame({
        "因子": ["默认参数", "最优参数", "合成因子"],
        "年化收益率": [default_perf["年化收益率"], best_perf["年化收益率"], synthesis_perf["年化收益率"]],
        "夏普比率": [default_perf["夏普比率"], best_perf["夏普比率"], synthesis_perf["夏普比率"]],
        "最大回撤": [default_perf["最大回撤"], best_perf["最大回撤"], synthesis_perf["最大回撤"]]
    })
    print(comparison.round(4))