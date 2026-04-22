import os
import pandas as pd
from datetime import datetime, timedelta

# 配置路径
LABEL_DATA_DIR = '/root/autodl-tmp/.autodl/project/data/daily_label_data'
FACTOR_DATA_DIR = '/root/autodl-tmp/.autodl/project/data/daily_factor_data'
OUTPUT_DIR = '/root/autodl-tmp/.autodl/project/data/trade_date_data'

# 滚动训练测试参数
TRAIN_PERIOD = 240          # 训练周期长度（日）
TEST_PERIOD = 30            # 测试周期长度（日）
ROLLING_STEP = 30           # 滚动周期（日）
GAP = 20                    # 数据泄露防护间隔（日）
BEGIN_DATE = '20150101'     # 开始日期（YYYYMMDD格式）


def extract_dates_from_files(directory):
    """从目录中提取所有CSV文件的日期"""
    dates = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # 提取日期部分（去掉.csv后缀）
            date_str = filename.replace('.csv', '')
            try:
                # 验证日期格式
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                # 检查是否在开始日期之后
                if date_obj < datetime.strptime(BEGIN_DATE, '%Y%m%d'):
                    continue
                dates.append(date_str)
            except ValueError:
                print(f"警告: 跳过无效日期格式的文件: {filename}")
                continue
    return dates


def get_common_dates(label_dates, factor_dates):
    """获取两个日期列表的交集并按时间排序"""
    # 转换为集合求交集
    label_set = set(label_dates)
    factor_set = set(factor_dates)
    common_dates = sorted(list(label_set & factor_set))
    return common_dates


def generate_rolling_periods(dates, train_period, test_period, rolling_step, gap):
    """
    生成滚动训练测试周期

    参数:
        dates: 排序后的日期列表
        train_period: 训练周期长度（日）
        test_period: 测试周期长度（日）
        rolling_step: 滚动步长（日）
        gap: 训练与测试之间的间隔（日）

    返回:
        periods: 列表，每个元素包含周期编号、训练日期列表、测试日期列表
    """
    periods = []
    n = len(dates)

    # 计算总需求长度：训练期 + 间隔 + 测试期
    total_needed = train_period + gap + test_period

    # 从第一个可用的训练周期开始
    start_idx = 0
    period_num = 1

    while start_idx + total_needed <= n:
        # 训练日期范围
        train_start_idx = start_idx
        train_end_idx = start_idx + train_period

        # 测试日期范围（训练结束后间隔gap天）
        test_start_idx = train_end_idx + gap
        test_end_idx = test_start_idx + test_period

        # 检查是否超出范围
        if test_end_idx > n:
            break

        # 提取训练和测试日期
        train_dates = dates[train_start_idx:train_end_idx]
        test_dates = dates[test_start_idx:test_end_idx]

        periods.append({
            'period_num': period_num,
            'train_dates': train_dates,
            'test_dates': test_dates,
            'train_start': train_dates[0],
            'train_end': train_dates[-1],
            'test_start': test_dates[0],
            'test_end': test_dates[-1]
        })

        # 滚动到下一个周期
        start_idx += rolling_step
        period_num += 1

    return periods


def save_period_to_csv(period, output_dir):
    """将单个周期的训练和测试日期保存到CSV文件"""
    period_num = period['period_num']

    # 创建训练日期DataFrame
    train_df = pd.DataFrame({
        'date': period['train_dates'],
        'type': ['train'] * len(period['train_dates'])
    })

    # 创建测试日期DataFrame
    test_df = pd.DataFrame({
        'date': period['test_dates'],
        'type': ['test'] * len(period['test_dates'])
    })

    # 合并训练和测试日期
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # 保存到CSV
    filename = f"period_{period_num:03d}_train_{period['train_start']}_{period['train_end']}_test_{period['test_start']}_{period['test_end']}.csv"
    filepath = os.path.join(output_dir, filename)
    combined_df.to_csv(filepath, index=False)

    return filename


def main():
    print("=" * 80)
    print("开始执行滚动训练测试日期切分任务")
    print("=" * 80)

    # 步骤1: 读取两个目录下的所有文件名称
    print("\n步骤1: 读取两个目录下的所有文件名称...")
    print(f"  目录1: {LABEL_DATA_DIR}")
    print(f"  目录2: {FACTOR_DATA_DIR}")

    label_dates = extract_dates_from_files(LABEL_DATA_DIR)
    factor_dates = extract_dates_from_files(FACTOR_DATA_DIR)

    print(f"  - daily_label_data 中的文件数: {len(label_dates)}")
    print(f"  - daily_factor_data 中的文件数: {len(factor_dates)}")

    # 步骤2: 提取并确定公共日期集合
    print("\n步骤2: 提取并确定公共日期集合...")
    common_dates = get_common_dates(label_dates, factor_dates)
    print(f"  - 公共日期数量: {len(common_dates)}")
    print(f"  - 最早日期: {common_dates[0]}")
    print(f"  - 最晚日期: {common_dates[-1]}")

    # 步骤3: 滚动训练测试日期切分
    print("\n步骤3: 执行滚动训练测试日期切分...")
    print(f"  - 训练周期长度: {TRAIN_PERIOD}日")
    print(f"  - 测试周期长度: {TEST_PERIOD}日")
    print(f"  - 滚动周期: {ROLLING_STEP}日")
    print(f"  - 数据泄露防护间隔(gap): {GAP}日")

    periods = generate_rolling_periods(
        common_dates,
        TRAIN_PERIOD,
        TEST_PERIOD,
        ROLLING_STEP,
        GAP
    )

    print(f"  - 生成的周期数量: {len(periods)}")

    # 步骤4 & 5: 保存每个周期的日期列表到CSV
    print("\n步骤4 & 5: 保存每个周期的日期列表到CSV文件...")
    print(f"  输出目录: {OUTPUT_DIR}")

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 保存汇总信息
    summary_data = []

    for period in periods:
        filename = save_period_to_csv(period, OUTPUT_DIR)
        summary_data.append({
            'period_num': period['period_num'],
            'train_start': period['train_start'],
            'train_end': period['train_end'],
            'train_count': len(period['train_dates']),
            'test_start': period['test_start'],
            'test_end': period['test_end'],
            'test_count': len(period['test_dates']),
            'filename': filename
        })

        print(f"  周期 {period['period_num']:03d}: "
              f"训练 [{period['train_start']} ~ {period['train_end']}] ({len(period['train_dates'])}日), "
              f"测试 [{period['test_start']} ~ {period['test_end']}] ({len(period['test_dates'])}日)")

    # 保存汇总表
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(OUTPUT_DIR, 'periods_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  汇总表已保存: {summary_path}")

    # 最终输出
    print("\n" + "=" * 80)
    print("任务完成!")
    print("=" * 80)
    print(f"\n处理结果:")
    print(f"  - 公共日期范围: {common_dates[0]} ~ {common_dates[-1]} (共{len(common_dates)}日)")
    print(f"  - 生成周期数: {len(periods)}")
    print(f"  - 每个周期: {TRAIN_PERIOD}日训练 + {GAP}日间隔 + {TEST_PERIOD}日测试")
    print(f"  - 输出目录: {OUTPUT_DIR}")

    # 验证日期逻辑
    print("\n日期逻辑验证:")
    for i, period in enumerate(periods[:3]):  # 显示前3个周期
        print(f"  周期 {period['period_num']:03d}:")
        print(f"    训练: {period['train_start']} ~ {period['train_end']} ({len(period['train_dates'])}日)")
        print(f"    间隔: {GAP}日")
        print(f"    测试: {period['test_start']} ~ {period['test_end']} ({len(period['test_dates'])}日)")

    if len(periods) > 3:
        print(f"  ... (共{len(periods)}个周期)")
        last_period = periods[-1]
        print(f"  周期 {last_period['period_num']:03d}:")
        print(f"    训练: {last_period['train_start']} ~ {last_period['train_end']} ({len(last_period['train_dates'])}日)")
        print(f"    间隔: {GAP}日")
        print(f"    测试: {last_period['test_start']} ~ {last_period['test_end']} ({len(last_period['test_dates'])}日)")


if __name__ == '__main__':
    main()
