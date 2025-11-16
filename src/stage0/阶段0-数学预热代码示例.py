"""
阶段0：数学直觉重启 - Python代码示例
目标：通过代码理解概率、统计的基本概念
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体（Windows）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ============ 1. 概率基础：抛硬币实验 ============
def coin_flip_experiment():
    """
    理解概率的频率定义和大数定律
    """
    print("=" * 50)
    print("实验1：抛硬币 - 理解概率")
    print("=" * 50)
    
    n_trials_list = [10, 100, 1000, 10000]
    
    for n in n_trials_list:
        # 模拟抛硬币（0=反面，1=正面）
        flips = np.random.randint(0, 2, size=n)
        heads_ratio = np.mean(flips)
        
        print(f"抛{n:5d}次，正面比例：{heads_ratio:.4f}")
    
    print("\n观察：次数越多，正面比例越接近0.5（理论概率）")
    print("这就是【大数定律】的直觉\n")


# ============ 2. 期望和方差：理解收益的特征 ============
def returns_statistics():
    """
    理解期望（平均收益）和方差（风险）
    """
    print("=" * 50)
    print("实验2：交易收益的期望和方差")
    print("=" * 50)
    
    # 模拟100次交易的收益率（%）
    np.random.seed(42)
    returns = np.random.normal(loc=0.5, scale=2.0, size=100)
    
    # 计算统计量
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    median_return = np.median(returns)
    
    print(f"平均收益（期望）：{mean_return:.2f}%")
    print(f"收益波动（标准差）：{std_return:.2f}%")
    print(f"中位数收益：{median_return:.2f}%")
    
    # 可视化
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(returns, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(mean_return, color='r', linestyle='--', label=f'均值={mean_return:.2f}%')
    plt.axvline(median_return, color='g', linestyle='--', label=f'中位数={median_return:.2f}%')
    plt.xlabel('收益率 (%)')
    plt.ylabel('频数')
    plt.title('收益率分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    cumulative_returns = np.cumsum(returns)
    plt.plot(cumulative_returns)
    plt.xlabel('交易次数')
    plt.ylabel('累计收益 (%)')
    plt.title('累计收益曲线')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('收益统计分析.png', dpi=100)
    print(f"\n图表已保存：收益统计分析.png")
    plt.close()


# ============ 3. 相关性：两个资产如何一起波动 ============
def correlation_example():
    """
    理解相关系数：BTC和ETH价格的关系
    """
    print("\n" + "=" * 50)
    print("实验3：相关性 - BTC与ETH的关系")
    print("=" * 50)
    
    # 模拟BTC和ETH的日收益率
    np.random.seed(42)
    n_days = 100
    
    # BTC收益率
    btc_returns = np.random.normal(0.2, 3.0, n_days)
    
    # ETH收益率：与BTC高度相关 + 自身波动
    eth_returns = 0.8 * btc_returns + np.random.normal(0.3, 1.5, n_days)
    
    # 计算相关系数
    correlation = np.corrcoef(btc_returns, eth_returns)[0, 1]
    
    print(f"BTC与ETH收益率的相关系数：{correlation:.3f}")
    print(f"解读：{correlation:.3f}表示两者{('强' if abs(correlation) > 0.7 else '中等' if abs(correlation) > 0.3 else '弱')}{'正' if correlation > 0 else '负'}相关")
    
    # 可视化
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(btc_returns, eth_returns, alpha=0.6)
    plt.xlabel('BTC 日收益率 (%)')
    plt.ylabel('ETH 日收益率 (%)')
    plt.title(f'BTC vs ETH 收益率散点图\n相关系数={correlation:.3f}')
    plt.grid(True, alpha=0.3)
    
    # 添加拟合线
    z = np.polyfit(btc_returns, eth_returns, 1)
    p = np.poly1d(z)
    plt.plot(btc_returns, p(btc_returns), "r--", alpha=0.8, label='拟合线')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(btc_returns, label='BTC', alpha=0.7)
    plt.plot(eth_returns, label='ETH', alpha=0.7)
    plt.xlabel('天数')
    plt.ylabel('收益率 (%)')
    plt.title('收益率时间序列对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('相关性分析.png', dpi=100)
    print(f"图表已保存：相关性分析.png\n")
    plt.close()


# ============ 4. 风险与收益权衡 ============
def risk_return_tradeoff():
    """
    理解为什么不能只看收益，还要看风险
    """
    print("=" * 50)
    print("实验4：风险与收益权衡")
    print("=" * 50)
    
    # 三种策略
    strategies = {
        '稳健策略': {'mean': 1.0, 'std': 2.0},
        '激进策略': {'mean': 2.0, 'std': 5.0},
        '极端策略': {'mean': 3.0, 'std': 10.0}
    }
    
    np.random.seed(42)
    n_days = 252  # 一年交易日
    
    results = {}
    for name, params in strategies.items():
        daily_returns = np.random.normal(params['mean'] / 252, params['std'] / np.sqrt(252), n_days)
        cumulative = np.cumprod(1 + daily_returns / 100) - 1
        results[name] = {
            'returns': daily_returns,
            'cumulative': cumulative,
            'sharpe': params['mean'] / params['std']  # 夏普比率简化版
        }
    
    # 输出结果
    for name, data in results.items():
        final_return = data['cumulative'][-1] * 100
        sharpe = data['sharpe']
        print(f"{name:8s}: 年收益={final_return:6.2f}%, 夏普比率={sharpe:.3f}")
    
    print("\n夏普比率：每承担1单位风险获得的收益")
    print("注意：收益最高的不一定是最好的策略！\n")
    
    # 可视化
    plt.figure(figsize=(10, 5))
    for name, data in results.items():
        plt.plot(data['cumulative'] * 100, label=name, linewidth=2)
    
    plt.xlabel('交易日')
    plt.ylabel('累计收益率 (%)')
    plt.title('不同风险策略的收益曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('风险收益权衡.png', dpi=100)
    print("图表已保存：风险收益权衡.png\n")
    plt.close()


# ============ 主函数 ============
if __name__ == "__main__":
    print("\n" + "🎯" * 25)
    print("阶段0：数学直觉代码示例")
    print("目标：通过代码理解量化交易的数学基础")
    print("🎯" * 25 + "\n")
    
    # 运行所有实验
    coin_flip_experiment()
    returns_statistics()
    correlation_example()
    risk_return_tradeoff()
    
    print("=" * 50)
    print("✅ 所有实验完成！")
    print("=" * 50)
    print("\n下一步：")
    print("1. 理解每个概念在量化交易中的作用")
    print("2. 尝试修改参数，观察结果变化")
    print("3. 准备学习真实的加密货币数据分析\n")
