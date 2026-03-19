"""
因子研究与回测脚本
- 读取收盘价宽表（zz500_close_wide.parquet）
- 定义多个跳跃类因子函数
- 对每个因子进行：计算、去极值、标准化、IC分析、多空回测
- 绘制净值曲线并与中证500指数对比
- 输出业绩指标（年化收益、夏普、最大回撤等）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  

# ==================== 因子定义 ====================
# 所有因子函数均以收盘价宽表（DataFrame，index为日期，columns为股票代码）为输入，
# 返回相同结构的因子值DataFrame。

def calc_jump_bipower(close, window=51):
    """
    基于BN-S双幂次变差思想的日频跳跃波动因子
    参数:
        close: pd.Series, 日频收盘价序列
        window: int, 滚动窗口长度（默认20天）
    返回:
        pd.Series, 每个交易日对应的跳跃波动估计值（非负）
    """
    ret = close.pct_change().dropna()  # 日收益率
    # 计算滚动窗口内的已实现方差 RV
    rv = (ret ** 2).rolling(window).sum()
    # 计算滚动窗口内的双幂次变差 BV (注意需要相邻收益率)
    # 方法：构造相邻收益率的绝对值乘积，然后滚动求和，乘以 pi/2
    abs_ret = ret.abs()
    bipower_core = abs_ret * abs_ret.shift(1)  # |r_t| * |r_{t-1}|
    bv = (np.pi / 2) * bipower_core.rolling(window).sum()
    # 跳跃波动 = max(RV - BV, 0)
    jump = (rv - bv).clip(lower=0)
    # 返回与close对齐的结果（前window天为NaN）
    return jump.reindex(close.index, fill_value=np.nan)



def calc_jump_momentum(close, window=42, momentum_window=15):
    """
    跳跃动量因子：跳跃日后的累计收益
    因子越大 = 跳跃后涨得越多 = 越可能高估
    """
    ret = close.pct_change()
    std_ret = ret.rolling(window).std()
    is_jump = ret.abs() > 2 * std_ret
    jump_ret = ret * is_jump
    return -jump_ret.rolling(momentum_window).sum()


def calc_jump_volatility_ratio(close, short_window=3, long_window=25):
    """
    波动率突变因子：短期波动率 / 长期波动率
    因子越大 = 近期波动突然放大 = 越可能高估
    """
    ret = close.pct_change()
    vol_short = ret.rolling(short_window).std()
    vol_long = ret.rolling(long_window).std()
    return vol_short / vol_long

def calc_jump_skewness(close, window=21):
    """
    收益偏度因子：正偏度表示右尾厚（向上跳跃多）
    因子越大 = 向上跳跃风险大 = 越可能高估
    """
    ret = close.pct_change()
    return -ret.rolling(window).skew()








# 将因子收集到一个字典中，方便循环调用
FACTORS = {
    "jump_bipower": calc_jump_bipower,
    "jump_momentum": calc_jump_momentum,
    "jump_volatility_ratio": calc_jump_volatility_ratio,
    "jump_skewness": calc_jump_skewness,
}

# ==================== 因子处理函数 ====================
def winsorize(factor, limits=[0.01, 0.01]):
    """
    MAD去极值：每期横截面，用中位数±3倍MAD截断
    """
    median = factor.median(axis=1)
    mad = (factor.subtract(median, axis=0)).abs().median(axis=1)
    upper = median + 3 * mad
    lower = median - 3 * mad
    return factor.clip(lower, upper, axis=0)

def standardize(factor):
    """
    横截面标准化：减去均值除以标准差，得到均值为0、标准差为1的序列
    """
    return factor.sub(factor.mean(axis=1), axis=0).div(factor.std(axis=1), axis=0)

# ==================== IC计算 ====================
def calc_ic(factor, forward_ret):
    """
    计算每日IC（因子与下一期收益的截面相关系数）
    factor, forward_ret均为DataFrame，索引日期，列股票
    返回IC序列（Series）
    """
    return factor.corrwith(forward_ret, axis=1, method='pearson')

# ==================== 回测函数 ====================
def run_backtest(factor, forward_ret, rebalance_freq='M',
                 start_date=None, end_date=None, initial_capital=1e6):
    """
    多空组合回测（基于因子偏离均值的幅度加权）

    Parameters
    ----------
    factor : pd.DataFrame
        因子值宽表，索引为日期，列为股票代码
    forward_ret : pd.DataFrame
        下一期收益宽表（t+1期收益），索引和列与factor一致
    rebalance_freq : str
        调仓频率：'D'每日，'W'每周，'M'每月
    start_date, end_date : str, optional
        回测起止日期，默认使用全部数据
    initial_capital : float
        初始资金

    Returns
    -------
    nav : pd.Series
        每日净值序列
    weights_history : dict
        每个调仓日的权重向量（可用于分析持仓）
    """
    # 统一频率大写
    rebalance_freq = rebalance_freq.upper()
    factor = factor.sort_index()
    forward_ret = forward_ret.sort_index()

    # 截取时间区间
    if start_date is not None:
        factor = factor.loc[start_date:]
        forward_ret = forward_ret.loc[start_date:]
    if end_date is not None:
        factor = factor.loc[:end_date]
        forward_ret = forward_ret.loc[:end_date]

    all_dates = factor.index
    if len(all_dates) == 0:
        raise ValueError("回测区间无数据")

    # 生成调仓日列表
    if rebalance_freq == 'D':
        rebalance_dates = all_dates
    elif rebalance_freq == 'W':
        week_starts = pd.date_range(start=all_dates[0], end=all_dates[-1], freq='W-MON')
        rebalance_dates = week_starts.intersection(all_dates)
    elif rebalance_freq == 'M':
        month_starts = pd.date_range(start=all_dates[0], end=all_dates[-1], freq='BMS')
        rebalance_dates = month_starts.intersection(all_dates)
    else:
        raise ValueError("rebalance_freq 必须为 'D', 'W' 或 'M'")

    # 构建调仓周期：每个调仓日到下一个调仓日前一天
    periods = []
    for i, d in enumerate(rebalance_dates):
        if i < len(rebalance_dates) - 1:
            end_d = rebalance_dates[i+1] - pd.Timedelta(days=1)
        else:
            end_d = all_dates[-1]
        hold_dates = all_dates[(all_dates > d) & (all_dates <= end_d)]
        periods.append((d, hold_dates))

    # 初始化每日持仓权重矩阵
    weights = pd.DataFrame(0.0, index=all_dates, columns=factor.columns)

    # 逐期计算权重并填充
    for rebalance_date, hold_dates in periods:
        factor_t = factor.loc[rebalance_date].dropna()
        if len(factor_t) == 0:
            continue
        mean_val = factor_t.mean()
        dev = factor_t - mean_val

        pos = dev[dev > 0]
        neg = dev[dev < 0]

        w = pd.Series(0.0, index=factor_t.index)
        if len(pos) > 0:
            w_pos = pos / pos.sum()          # 多头按正偏离比例分配，权重和为1
            w.loc[pos.index] = w_pos
        if len(neg) > 0:
            w_neg = neg / neg.sum()          # 注意：neg为负，neg.sum()为负，所以w_neg为正
            w.loc[neg.index] = -w_neg        # 修正：空头权重设为负数，实现多空市值相等

        if len(hold_dates) > 0:
            weights.loc[hold_dates, w.index] = w.values

    # 计算每日组合收益（当日收盘建仓，下日收益）
    weights_aligned = weights.shift(1).fillna(0)
    daily_ret = (weights_aligned * forward_ret).sum(axis=1)
    daily_ret = daily_ret.loc[forward_ret.index]   # 对齐日期

    # 计算净值
    nav = initial_capital * (1 + daily_ret).cumprod()

    # 记录调仓日权重
    weights_history = {date: weights.loc[date].dropna() for date in rebalance_dates}

    return nav, weights_history

# ==================== 业绩指标 ====================
def compute_performance(nav, risk_free_rate=0.03):
    """
    计算年化收益率、年化波动率、夏普比率、最大回撤
    nav : pd.Series, 净值序列
    risk_free_rate : float, 年化无风险利率
    """
    ret = nav.pct_change().dropna()
    total_days = len(nav)
    years = total_days / 252
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    ann_return = (1 + total_return) ** (1 / years) - 1

    ann_vol = ret.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol

    # 最大回撤
    peak = nav.expanding().max()
    drawdown = (nav - peak) / peak
    max_dd = drawdown.min()

    return {
        "年化收益率": ann_return,
        "年化波动率": ann_vol,
        "夏普比率": sharpe,
        "最大回撤": max_dd
    }

# ==================== 可视化 ====================
def plot_navs(nav_dict, index_norm=None, title="多因子净值对比"):
    """
    绘制多个因子净值曲线
    nav_dict : dict, key为因子名称，value为净值Series
    index_norm : Series, 基准指数归一化净值
    """
    plt.figure(figsize=(12, 6))
    for name, nav in nav_dict.items():
        nav_norm = nav / nav.iloc[0]
        plt.plot(nav_norm.index, nav_norm, label=name, linewidth=2)

    if index_norm is not None:
        plt.plot(index_norm.index, index_norm, label='中证500', linewidth=2, linestyle='--', color='black')

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
    plt.title(title)
    plt.xlabel('日期')
    plt.ylabel('累计收益率')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==================== 主流程 ====================
if __name__ == "__main__":
    # 1. 读取数据
    print("读取收盘价宽表...")
    close_wide = pd.read_parquet("zz500_close_wide.parquet", engine='fastparquet')
    close_wide.index = pd.to_datetime(close_wide.index)
    close_wide.sort_index(inplace=True)

    # 读取指数数据（用于对比）
    try:
        index_df = pd.read_parquet("zz500_index.parquet", engine='fastparquet')
        index_close = index_df['close'].sort_index()
    except:
        print("⚠️ 未找到指数文件，将跳过基准对比")
        index_close = None

    # 2. 计算下一期收益（t+1期收益，用于IC和回测）
    forward_ret = close_wide.pct_change().shift(-1)

    # 3. 设定回测区间（可根据需要调整）
    bt_start = "2015-01-01"
    bt_end = "2021-12-31"

    # 4. 存储每个因子的净值
    navs = {}
    performance_records = []

    # 5. 对每个因子循环计算
    for name, func in FACTORS.items():
        print(f"\n========== 处理因子: {name} ==========")

        # 计算因子值
        factor_raw = func(close_wide)

        # 删除全为NaN的列（股票可能退市或未上市）
        factor_raw = factor_raw.dropna(axis=1, how='all')

        # 因子处理：去极值、标准化
        factor_processed = winsorize(factor_raw)
        factor_processed = standardize(factor_processed)

        # 对齐数据：剔除因子或收益中缺失的日期
        common_idx = factor_processed.index.intersection(forward_ret.index)
        factor_processed = factor_processed.loc[common_idx]
        fwd_ret_aligned = forward_ret.loc[common_idx]

        # 计算IC
        ic = calc_ic(factor_processed, fwd_ret_aligned)
        ic_mean = ic.mean()
        ic_std = ic.std()
        ir = ic_mean / ic_std if ic_std != 0 else np.nan
        print(f"IC mean: {ic_mean:.4f}, IC std: {ic_std:.4f}, IR: {ir:.4f}")

        # 运行回测
        try:
            nav, _ = run_backtest(
                factor_processed, fwd_ret_aligned,
                rebalance_freq='M',
                start_date=bt_start,
                end_date=bt_end
            )
            # 确保nav为float
            nav = pd.to_numeric(nav, errors='coerce').dropna()
            if len(nav) == 0:
                print("⚠️ 回测未产生有效净值")
                continue
            navs[name] = nav

            # 计算业绩指标
            perf = compute_performance(nav)
            perf['因子'] = name
            performance_records.append(perf)
            print(f"年化收益: {perf['年化收益率']:.2%}, 夏普: {perf['夏普比率']:.2f}, 最大回撤: {perf['最大回撤']:.2%}")
        except Exception as e:
            print(f"⚠️ 回测失败: {e}")
            continue

    # 6. 输出所有因子业绩对比表
    if performance_records:
        perf_df = pd.DataFrame(performance_records)
        perf_df = perf_df.set_index('因子')
        print("\n========== 因子业绩汇总 ==========")
        print(perf_df.round(4))

    # 7. 绘制净值曲线（全部因子 + 基准）
    if navs:
        # 归一化基准
        if index_close is not None:
            common_dates = index_close.index.intersection(navs[list(navs.keys())[0]].index)
            index_norm = index_close.loc[common_dates] / index_close.loc[common_dates].iloc[0]
        else:
            index_norm = None

        plot_navs(navs, index_norm, title="跳跃类因子多空组合净值对比")
    else:
        print("没有因子成功运行回测。")