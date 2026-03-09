"""
中证500数据获取脚本
- 从Wind获取指数日线行情（000905.SH）
- 获取指定日期成分股列表（此处以2011-01-01为例，实际使用中需动态获取）
- 下载所有成分股日线行情（长表）
- 保存原始长表及收盘价、换手率宽表（Parquet格式）
"""

import pandas as pd
import numpy as np
from WindPy import w
from datetime import datetime
import time
import os
from tqdm import tqdm

# ==================== Wind连接 ====================
def init_wind():
    """启动Wind接口，确保连接成功"""
    try:
        w.start()
        if w.isconnected():
            print("✅ Wind连接成功")
            return True
        else:
            print("❌ Wind连接失败，请检查终端是否已登录")
            return False
    except Exception as e:
        print(f"❌ Wind初始化异常: {e}")
        return False

# ==================== 数据获取函数 ====================
def fetch_index_data(code="000905.SH", start_date="2011-01-01", end_date="2026-03-08"):
    """
    获取指数日线行情
    返回DataFrame，索引为日期，字段：open,high,low,close,volume,amount,pct_chg,turn,swing
    """
    print(f"正在获取 {code} 日线数据: {start_date} 至 {end_date}")
    fields = "open,high,low,close,volume,amt,pct_chg,turn,swing"
    data = w.wsd(code, fields, start_date, end_date, "Fill=Previous")

    if data.ErrorCode != 0:
        print(f"❌ 数据获取失败，错误码: {data.ErrorCode}")
        return None

    df = pd.DataFrame(
        data.Data,
        index=["open", "high", "low", "close", "volume", "amount", "pct_chg", "turn", "swing"],
        columns=data.Times
    ).T
    df.index.name = "date"
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["return"] = df["close"].pct_change()          # 方便回测使用
    print(f"✅ 获取成功，共 {len(df)} 条记录")
    return df

def get_constituents(date="2011-01-01"):
    """获取指定日期中证500成分股列表"""
    res = w.wset("indexconstituent", f"date={date};windcode=000905.SH")
    if res.ErrorCode != 0:
        print(f"❌ 获取成分股失败，错误码: {res.ErrorCode}")
        return None
    codes = res.Data[1]   # Wind代码
    names = res.Data[2]   # 股票名称
    df = pd.DataFrame({"code": codes, "name": names})
    print(f"✅ 获取到 {len(df)} 只成分股 (参考日期: {date})")
    return df

def fetch_stocks_daily(stock_list, start_date="2011-01-01", end_date="2026-03-08"):
    """
    逐个下载股票日线行情（较慢但稳妥）
    返回长表，包含字段：date, code, open,high,low,close,volume,amount,pct_chg,turn,swing
    """
    fields = "open,high,low,close,volume,amt,pct_chg,turn,swing"
    all_data = []
    total = len(stock_list)
    print(f"开始逐个下载 {total} 只股票数据...")

    for i, code in enumerate(tqdm(stock_list, desc="下载进度")):
        data = w.wsd(code, fields, start_date, end_date, "Fill=Previous;PriceAdj=F")
        if data.ErrorCode == 0:
            df_stock = pd.DataFrame(
                data.Data,
                index=fields.split(','),
                columns=data.Times
            ).T
            df_stock['code'] = code
            df_stock = df_stock.reset_index().rename(columns={'index': 'date'})
            all_data.append(df_stock)
        else:
            print(f"\n⚠️ 股票 {code} 下载失败，错误码: {data.ErrorCode}，跳过")
        time.sleep(0.1)   # 避免请求过快

    if not all_data:
        print("❌ 未获取到任何数据")
        return None

    result = pd.concat(all_data, ignore_index=True)
    result['date'] = pd.to_datetime(result['date'])
    result = result.sort_values(['date', 'code']).reset_index(drop=True)
    print(f"✅ 数据下载完成，成功获取 {result['code'].nunique()} 只股票")
    return result

# ==================== 保存函数 ====================
def save_parquet(df, filename, index=False):
    """保存DataFrame为Parquet文件（使用fastparquet引擎，snappy压缩）"""
    df.to_parquet(filename, engine='fastparquet', compression='snappy')
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"💾 已保存: {filename} ({size_mb:.2f} MB)")

def pivot_to_wide(df, value_col):
    """将长表转换为宽表：日期为索引，股票代码为列"""
    return df.pivot_table(index='date', columns='code', values=value_col)

# ==================== 主流程 ====================
if __name__ == "__main__":
    # 1. 连接Wind
    if not init_wind():
        exit()

    # 2. 获取指数数据并保存
    index_df = fetch_index_data(start_date="2011-01-01", end_date="2026-03-08")
    if index_df is not None:
        save_parquet(index_df, "zz500_index.parquet", index=True)

    # 3. 获取成分股列表（以2011-01-01为例，实际可扩展为动态获取）
    constituents_df = get_constituents(date="2011-01-01")
    if constituents_df is None:
        exit()
    # 保存成分股列表（可选）
    constituents_df.to_csv("zz500_constituents_20110101.csv", index=False)

    # 4. 下载所有成分股日线行情
    stock_list = constituents_df['code'].tolist()
    stocks_long = fetch_stocks_daily(stock_list, start_date="2011-01-01", end_date="2026-03-08")
    if stocks_long is None:
        exit()

    # 5. 保存原始长表（作为备份）
    save_parquet(stocks_long, "zz500_stocks_long.parquet", index=False)

    # 6. 生成并保存宽表（收盘价、换手率）
    print("生成收盘价宽表...")
    close_wide = pivot_to_wide(stocks_long, 'close')
    save_parquet(close_wide, "zz500_close_wide.parquet", index=True)

    print("生成换手率宽表...")
    turnover_wide = pivot_to_wide(stocks_long, 'turn')
    save_parquet(turnover_wide, "zz500_turnover_wide.parquet", index=True)

    # 7. 输出统计信息
    print("\n" + "="*60)
    print("📊 数据统计:")
    print(f"  股票数量: {stocks_long['code'].nunique()}")
    print(f"  日期范围: {stocks_long['date'].min()} 至 {stocks_long['date'].max()}")
    print(f"  交易日数量: {stocks_long['date'].nunique()}")
    print(f"  总记录数: {len(stocks_long):,}")
    print("="*60)