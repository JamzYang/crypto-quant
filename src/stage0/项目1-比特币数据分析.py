"""
项目1：比特币价格数据分析
目标：
1. 获取BTC历史数据
2. 计算日收益率和统计量
3. 绘制价格曲线和收益率分布
4. 计算最大回撤
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def fetch_binance_data(symbol='BTCUSDT', interval='1d', days=365):
    """
    从币安获取历史K线数据
    
    参数:
        symbol: 交易对，如'BTCUSDT'
        interval: K线周期，如'1d'(日线),'1h'(小时线)
        days: 获取最近多少天的数据
    """
    try:
        import ccxt
        
        exchange = ccxt.binance()
        
        # 计算开始时间
        since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
        
        # 获取K线数据
        ohlcv = exchange.fetch_ohlcv(symbol, interval, since=since)
        
        # 转换为DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"✅ 成功获取 {symbol} 数据，共 {len(df)} 条记录")
        print(f"时间范围：{df.index[0]} 至 {df.index[-1]}\n")
        
        return df
    
    except ImportError:
        print("⚠️ 未安装ccxt库，使用模拟数据")
        print("提示：运行 pip install ccxt 安装\n")
        return generate_mock_data(days)


def generate_mock_data(days=365):
    """
    生成模拟的BTC价格数据（用于演示）
    """
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # 模拟价格随机游走
    returns = np.random.normal(0.001, 0.03, days)
    price = 30000 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'open': price * (1 + np.random.uniform(-0.01, 0.01, days)),
        'high': price * (1 + np.random.uniform(0, 0.02, days)),
        'low': price * (1 + np.random.uniform(-0.02, 0, days)),
        'close': price,
        'volume': np.random.uniform(1000, 5000, days)
    }, index=dates)
    
    return df


def calculate_returns(df):
    """
    计算收益率
    """
    # 简单收益率
    df['simple_return'] = df['close'].pct_change()
    
    # 对数收益率
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    return df


def calculate_statistics(df):
    """
    计算基本统计量
    """
    returns = df['simple_return'].dropna()
    
    stats = {
        '平均日收益率': returns.mean(),
        '收益率标准差': returns.std(),
        '年化收益率': returns.mean() * 252,
        '年化波动率': returns.std() * np.sqrt(252),
        '最大单日涨幅': returns.max(),
        '最大单日跌幅': returns.min(),
        '正收益天数比例': (returns > 0).mean()
    }
    
    return stats


def calculate_max_drawdown(df):
    """
    计算最大回撤
    最大回撤 = (谷底价格 - 峰顶价格) / 峰顶价格
    """
    cumulative = (1 + df['simple_return']).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    
    return max_dd, max_dd_date, drawdown


def plot_analysis(df, stats, drawdown):
    """
    绘制综合分析图表
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('比特币价格分析报告', fontsize=16, fontweight='bold')
    
    # 1. 价格曲线
    ax1 = axes[0, 0]
    ax1.plot(df.index, df['close'], linewidth=2, color='#1f77b4')
    ax1.set_title('BTC价格走势', fontsize=12, fontweight='bold')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('价格 (USDT)')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. 收益率分布
    ax2 = axes[0, 1]
    returns = df['simple_return'].dropna() * 100
    ax2.hist(returns, bins=50, alpha=0.7, color='#ff7f0e', edgecolor='black')
    ax2.axvline(returns.mean(), color='r', linestyle='--', linewidth=2, label=f'均值={returns.mean():.3f}%')
    ax2.set_title('日收益率分布', fontsize=12, fontweight='bold')
    ax2.set_xlabel('收益率 (%)')
    ax2.set_ylabel('频数')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 回撤曲线
    ax3 = axes[1, 0]
    ax3.fill_between(df.index, drawdown * 100, 0, alpha=0.3, color='red')
    ax3.plot(df.index, drawdown * 100, linewidth=2, color='darkred')
    ax3.set_title('回撤曲线', fontsize=12, fontweight='bold')
    ax3.set_xlabel('日期')
    ax3.set_ylabel('回撤 (%)')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. 统计信息表格
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = []
    for key, value in stats.items():
        if '比例' in key:
            table_data.append([key, f'{value*100:.2f}%'])
        else:
            table_data.append([key, f'{value*100:.3f}%'])
    
    table = ax4.table(cellText=table_data, colLabels=['指标', '数值'],
                      cellLoc='left', loc='center',
                      colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置交替行颜色
    for i in range(1, len(table_data) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax4.set_title('统计指标', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('BTC价格分析报告.png', dpi=150, bbox_inches='tight')
    print("\n📊 分析图表已保存：BTC价格分析报告.png")


def main():
    """
    主函数
    """
    print("\n" + "="*60)
    print("项目1：比特币价格数据分析")
    print("="*60 + "\n")
    
    # 1. 获取数据
    print("📥 正在获取数据...")
    df = fetch_binance_data(symbol='BTCUSDT', interval='1d', days=365)
    
    # 2. 计算收益率
    print("📊 计算收益率...")
    df = calculate_returns(df)
    
    # 3. 计算统计量
    print("📈 计算统计指标...")
    stats = calculate_statistics(df)
    
    # 4. 计算最大回撤
    print("📉 计算最大回撤...")
    max_dd, max_dd_date, drawdown = calculate_max_drawdown(df)
    
    # 5. 输出结果
    print("\n" + "="*60)
    print("📋 分析结果")
    print("="*60)
    
    for key, value in stats.items():
        if '比例' in key:
            print(f"{key:15s}: {value*100:7.2f}%")
        else:
            print(f"{key:15s}: {value*100:7.3f}%")
    
    print(f"\n最大回撤: {max_dd*100:.2f}% (发生在 {max_dd_date.strftime('%Y-%m-%d')})")
    
    # 6. 绘制图表
    print("\n📊 生成分析图表...")
    plot_analysis(df, stats, drawdown)
    
    # 7. 保存数据
    df.to_csv('btc_data.csv')
    print("💾 数据已保存：btc_data.csv")
    
    print("\n" + "="*60)
    print("✅ 分析完成！")
    print("="*60)
    
    print("\n💡 下一步：")
    print("1. 分析不同时间周期的数据（周线、月线）")
    print("2. 对比BTC和ETH的统计特征")
    print("3. 研究价格与成交量的关系\n")


if __name__ == "__main__":
    main()
