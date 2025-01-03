import logging
import os
from datetime import datetime, timedelta
from enum import Enum

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SEABORN_STYLE = "whitegrid"
FONT_SANS_SERIF = "Heiti TC"
AXES_UNICODE_MINUS = False

FIG_SIZE = (12, 6)
ROTATION_ANGLE = 45
TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 14
LEGEND_LINE_LABEL = "总数"

TITLE_MONTHLY = "按月聚合"
XLABEL_MONTHLY = "月份"
TITLE_QUARTERLY = "按季度聚合"
XLABEL_QUARTERLY = "季度"
TITLE_YEARLY = "按年聚合"
XLABEL_YEARLY = "年份"

xiaoke_table_path = "/Users/luxun/Documents/2025消课表.xlsx"
summary_table_path = "/Users/luxun/Desktop/学员上课缴费汇总表.xlsx"
baoke_table_path = "/Users/luxun/Documents/2025报课表.xlsx"

# 设置日志配置
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


def aggregate_data(df: pd.DataFrame, time_col: str, group_col: str, y_col: str, y_col2: str = None, radio: bool = False) -> pd.DataFrame:
    """
    通用聚合函数：对 DataFrame 进行按 (time_col, group_col) 分组，
    汇总数值列 y_col
    返回结果是一个 pivot 表，
    行索引= time_col (各时间段)，列= group_col 的不同取值，
    单元格= y_col 汇总后的万元值。
    """
    if y_col2:
        grouped = (
            df.groupby([time_col, group_col])[[y_col, y_col2]]
            .sum()
            .reset_index()
        )
        # 计算 y_col1 和 y_col2 的比率
        grouped['ratio'] = grouped[y_col] / grouped[y_col2]
        pivot_table = grouped.pivot(index=time_col, columns=group_col, values='ratio').fillna(0)
        return pivot_table
    else:
        grouped = (
            df.groupby([time_col, group_col])[y_col]
            .sum()
            .reset_index()
        )

        if radio:
            # 计算每个组的计数
            count = df.groupby([time_col, group_col])[y_col].count().reset_index(name='count')
            # 合并分组的总和和计数
            grouped = grouped.merge(count, on=[time_col, group_col])

            # 计算每个组的平均值
            grouped[y_col] = (grouped[y_col] / (grouped['count'] * 4)).round(2)
            # 删除不再需要的 'count' 列
            grouped = grouped.drop(columns=['count'])
        else:
            grouped[y_col] = (grouped[y_col]).round(2)

        pivot_table = grouped.pivot(index=time_col, columns=group_col, values=y_col).fillna(0)

        return pivot_table


def plot_two_bar_plus_line(data: pd.DataFrame, title: str, xlabel: str, ylabel: str, show_zhexian: bool = True):
    """
    绘制并排柱（data各列）+ 叠加一条“总和”折线。
    默认认为 data 有两列（如“首次”、“续费”），也可扩展为更多列。
    """
    # 1) 绘制并排柱状图
    ax = data.plot(kind="bar", figsize=FIG_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    plt.xticks(rotation=ROTATION_ANGLE)

    # 2) 为每根柱子标数值 (Matplotlib >= 3.4)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type='edge', padding=3)

    if show_zhexian:
        # 3) 叠加折线图：对每行求和得到“总和”
        total_series = data.sum(axis=1).round(4)
        x_positions = range(len(data))  # 与柱状图索引对齐

        ax.plot(
            x_positions,
            total_series.values,
            marker='o',
            color='red',
            linewidth=2,
            label=LEGEND_LINE_LABEL
        )
        # 4) 在折线上标注数值
        for x_idx, val in enumerate(total_series.values):
            ax.annotate(
                f"{val:.2f}",
                (x_idx, val),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color='red',
                fontsize=10
            )

    ax.legend()  # 显示图例（包括柱的列名 & 折线label）
    plt.tight_layout()
    plt.show()


def process_excel(
        excel_file: str,
        y_col: str,
        time_col: str,
        group_col: str,
        y_label: str,
        y_col2: str = None,
        radio: bool = False,
        scope=None

):
    """
    通用处理流程：
    1. 读取Excel
    2. 将 time_col 转为 datetime，并分别生成 Year/Quarter/Month 列
    3. 分别聚合: 按月、季度、年
    4. 绘图：并排柱 + 折线
    """
    # 1) 读取 Excel

    if scope is None:
        scope = ['Month', 'Quarter', 'Year']
    if not os.path.exists(excel_file):
        raise FileNotFoundError(f"找不到文件: {excel_file}")
    df = pd.read_excel(excel_file)

    # 2) 转换时间列
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')  # 若格式不统一可加 format=
    df["Year"] = df[time_col].dt.year.astype("Int64")  # 为了避免NaN导致float，可用Int64
    df["Quarter"] = df[time_col].dt.to_period('Q').astype(str)
    df["Month"] = df[time_col].dt.to_period('M').astype(str)

    # 设置全局风格 (可放到外面做一次即可)
    sns.set(
        style=SEABORN_STYLE,
        rc={
            "font.sans-serif": [FONT_SANS_SERIF],
            "axes.unicode_minus": AXES_UNICODE_MINUS
        }
    )

    # 1) 定义一个 scope 列表，包含不同的时间范围

    # 2) 遍历 scope 列表进行聚合和绘图
    for period in scope:
        # 聚合数据
        aggregated_df = aggregate_data(df, period, group_col, y_col, y_col2, radio)
        # 绘图
        if period == 'Month':
            plot_two_bar_plus_line(aggregated_df, TITLE_MONTHLY, XLABEL_MONTHLY, y_label, not radio)
        elif period == 'Quarter':
            plot_two_bar_plus_line(aggregated_df, TITLE_QUARTERLY, XLABEL_QUARTERLY, y_label, not radio)
        elif period == 'Year':
            plot_two_bar_plus_line(aggregated_df, TITLE_YEARLY, XLABEL_YEARLY, y_label, not radio)


def draw_graph():
    # 消课率指标
    process_excel(
        excel_file=xiaoke_table_path,
        y_col="消课节数",
        time_col="消课时间",
        group_col="消课年龄分布",
        y_label="消课率(%)",
        radio=True,
        scope=['Month', 'Quarter', 'Year']
    )
    # 消课节数
    process_excel(
        excel_file=xiaoke_table_path,
        y_col="消课节数",
        time_col="消课时间",
        group_col="消课年龄分布",
        y_label="消课节数",
        radio=False,
        scope=['Month', 'Quarter', 'Year']
    )
    # 消课金额指标
    process_excel(
        excel_file=xiaoke_table_path,
        y_col="消课金额",
        time_col="消课时间",
        group_col="消课年龄分布",
        y_label="消课金额",
        radio=False,
        scope=['Month', 'Quarter', 'Year']
    )

    process_excel(
        excel_file=baoke_table_path,
        y_col="待消金额",
        time_col="报课时间",
        group_col="年龄分布",
        y_label="待消金额（按报课时间聚合）",
        radio=False,
        scope=['Quarter', 'Year']
    )

    process_excel(
        excel_file=baoke_table_path,
        y_col="待消节数",
        time_col="报课时间",
        group_col="年龄分布",
        y_label="待消节数（按报课时间聚合）",
        radio=False,
        scope=['Quarter', 'Year']
    )

    # 续费率指标
    # process_excel(
    #     excel_file=summary_table_path,
    #     y_col="实际续费人数",
    #     y_col2="应续费人数",
    #     time_col="统计时间",
    #     group_col="年龄分布",
    #     y_label="续费率(%)",
    #     radio=True
    # )
    # 缴费金额指标
    process_excel(
        excel_file=baoke_table_path,
        y_col="报课总费用",
        time_col="报课时间",
        group_col="年龄分布",
        y_label="缴费金额（按年龄分布）",
        radio=False,
        scope=['Quarter', 'Year']
    )
    process_excel(
        excel_file=baoke_table_path,
        y_col="报课总费用",
        time_col="报课时间",
        group_col="报课类别",
        y_label="缴费金额（按首次和续费）",
        radio=False,
        scope=['Quarter', 'Year']
    )


class CourseState(Enum):
    IN_PROCESS = '消课中'
    DONE = '已消课'
    IN_PAUSE = '暂停中'
    CANCEL = "已取消"


class FeeType(Enum):
    FEE_TYPE_FIRST = '首次'
    FEE_TYPE_RENEW = '续费'


# 计算应上课人数
def calc_people_number(month: str):
    df = pd.read_excel(baoke_table_path)
    # 应上课人数
    filtered_df = df[df['消课状态'] == CourseState.IN_PROCESS.value]
    count_in_process = filtered_df.shape[0]

    # 应续费人数
    first_day_of_month, last_day_of_last_month = get_month_first_last_day(month)
    filtered_df = df[(df['消课状态'] == CourseState.DONE.value)
                     & (df['消课结束时间'] >= first_day_of_month)
                     & (df['消课结束时间'] <= last_day_of_last_month)]
    count_done_last_month = filtered_df.shape[0]

    # 实际续费人数
    filtered_df = df[(df['报课类别'] == FeeType.FEE_TYPE_RENEW.value)
                     & (df['报课时间'] >= first_day_of_month)
                     & (df['报课时间'] <= last_day_of_last_month)]
    count_real_renewal = filtered_df.shape[0]

    # 实际上课人数
    df = pd.read_excel(xiaoke_table_path)
    filtered_df = df[(df['消课时间'] >= first_day_of_month) &
                     (df['消课时间'] <= last_day_of_last_month)]
    count_real_class = filtered_df.shape[0]

    return {
        "count_in_process": count_in_process,
        "count_real_class": count_real_class,
        "count_done_last_month": count_done_last_month,
        "count_real_renewal": count_real_renewal
    }


def get_month_first_last_day(time_str: str):
    # '2024-12-01'
    date_time = datetime.strptime(time_str, "%Y-%m-%d")

    # 计算一个月的第一天并设置为00:00:00
    first_day_of_month = pd.Timestamp(date_time.replace(day=1))
    first_day_of_month = first_day_of_month.replace(hour=0, minute=0, second=0, microsecond=0)

    # 计算一个月的最后一天并设置为23:59:59
    last_day_of_month = first_day_of_month + pd.DateOffset(months=1) - timedelta(days=1)
    last_day_of_month = last_day_of_month.replace(hour=23, minute=59, second=59, microsecond=999999)

    return first_day_of_month, last_day_of_month


# 年龄分布列根据年级列计算得出
def calculate_age_group(grade):
    if grade in ['一年级', '二年级']:
        return '小学低年级'
    elif grade in ['三年级', '四年级', '五年级']:
        return '小学高年级'
    elif grade in ['小班', '中班', '大班', '幼儿园']:
        return '幼儿园'
    else:
        return '初中'


def update_baoke_table_base_info(is_force: bool = False):
    data = pd.read_excel(baoke_table_path)
    if is_force:
        data['年龄分布'] = data['年级'].apply(calculate_age_group)
        # 课程单价为报课总费用除以报课节数，向上取整
        data['课程单价'] = (data['报课总费用'] / data['报课节数']).round(1)
        data.loc[:, '待消节数'] = data['报课节数'] - data['已消节数'].fillna(0)
        data.loc[:, '最后消课时间'] = pd.Timestamp('2024-12-31')
        data.loc[:, '已消金额'] = data['课程单价'] * data['已消节数']
        data.loc[:, '待消金额'] = data['报课总费用'] - data['已消金额']
    else:
        # 仅在'年龄分布'为空时更新
        data['年龄分布'] = data.apply(
            lambda row: calculate_age_group(row['年级']) if pd.isna(row['年龄分布']) else row['年龄分布'], axis=1
        )

        # 仅在'课程单价'为空时更新
        data['课程单价'] = data.apply(
            lambda row: (row['报课总费用'] / row['报课节数']).round(1) if pd.isna(row['课程单价']) else row['课程单价'], axis=1
        )
    # 待消节数为报课节数减去已消节数
    data.to_excel(baoke_table_path, index=False)


def update_all_table(month: str, is_force: bool = False):
    update_baoke_table_base_info(is_force)
    update_xiaoke_table_base_info()
    # update_baoke_table_xiaoke_state(month)


def update_baoke_table_xiaoke_state(month: str):
    # 读取报课表和消课表
    baoke_data = pd.read_excel(baoke_table_path)
    xiaoke_data = pd.read_excel(xiaoke_table_path)

    first_day_of_month, last_day_of_month = get_month_first_last_day(month)
    xiaoke_data = xiaoke_data[(xiaoke_data['消课时间'] >= first_day_of_month)
                              & (xiaoke_data['消课时间'] <= last_day_of_month)]

    # 关联两个表，基于报课编号
    data_merged = baoke_data.merge(xiaoke_data[['报课编号', '消课节数', '消课时间']], on='报课编号', how='left')
    data_merged = data_merged.query('消课状态 == "消课中"')
    assert data_merged.size > 0, "没有消课中的单子！！！"

    # 如果为空先填充一个值
    data_merged.loc[:, '最后消课时间'] = data_merged['最后消课时间'].fillna(pd.Timestamp('1970-01-01'))
    data_merged = data_merged[data_merged['最后消课时间'] < data_merged['消课时间']]
    if data_merged.size <= 0:
        logging.warning("当前月份已完成全部消课，无须重复消课!!!")
        return

    def check_status(x):
        if x < 0:
            raise ValueError(f"待消金额不能小于0，遇到值: {x}")
        return '已消课' if x == 0 else '消课中'

    # 使用 .loc 来避免 SettingWithCopyWarning
    data_merged.loc[:, '已消节数'] = data_merged['已消节数'] + data_merged['消课节数'].fillna(0)
    data_merged.loc[:, '待消节数'] = data_merged['报课节数'] - data_merged['已消节数'].fillna(0)
    data_merged.loc[:, '最后消课时间'] = data_merged['消课时间']
    data_merged.loc[:, '已消金额'] = data_merged['课程单价'] * data_merged['已消节数']
    data_merged.loc[:, '待消金额'] = data_merged['报课总费用'] - data_merged['已消金额']
    data_merged.loc[:, '消课状态'] = data_merged['待消金额'].apply(check_status)

    # 消课状态列如果待消金额为0，则消课状态取值”消课完成“，否则保持原
    data_merged.drop(columns=['消课节数', '消课时间'], inplace=True)
    if data_merged['报课编号'].duplicated().any():
        print("警告：报课编号存在重复项！")
        raise ValueError("报课编号存在重复项！")

    # 读取原始数据并与更新的数据合并，保留所有行，更新变更的行
    updated_data = baoke_data.copy()
    # 更新变更行
    updated_data.update(data_merged)
    updated_data.to_excel(baoke_table_path, index=False)


def update_xiaoke_table_base_info():
    # 读取报课表和消课表
    baoke_data = pd.read_excel(baoke_table_path)
    xiaoke_data = pd.read_excel(xiaoke_table_path)
    # 关联两个表，基于报课编号
    data_merged = xiaoke_data.merge(baoke_data[['报课编号', '课程单价', '年龄分布']], on='报课编号', how='left')

    # 只更新消课金额为空的行
    data_merged.loc[data_merged['消课金额'].isna(), '消课金额'] = data_merged['课程单价'] * data_merged['消课节数']

    # 将消课单价列取自课程单价，若消课单价为空则使用课程单价
    data_merged['消课单价'] = data_merged['消课单价'].fillna(data_merged['课程单价'])

    data_merged['消课年龄分布'] = data_merged['消课年龄分布'].fillna(data_merged['年龄分布'])

    # 删除合并后的课程单价列
    data_merged.drop(columns=['课程单价', '年龄分布'], inplace=True)

    # 保存处理后的数据到本地
    data_merged.to_excel(xiaoke_table_path, index=False)


def update_student_summary_table(month: str):
    df = pd.read_excel(summary_table_path)
    count = calc_people_number(month)
    _, last_day_of_month = get_month_first_last_day(month)
    new_record = {
        '应上课人数': count["count_in_process"],
        '实际上课人数': count["count_real_class"],
        '应续费人数': count["count_done_last_month"],
        '实际续费人数': count["count_real_renewal"],
        '统计时间': last_day_of_month.strftime('%Y-%m-%d')
    }

    # 将新记录转换为 DataFrame
    new_record_df = pd.DataFrame([new_record])
    # 使用 pd.concat() 追加新记录
    df = pd.concat([df, new_record_df], ignore_index=True)

    # 将更新后的 DataFrame 保存回 Excel 文件
    df.to_excel(summary_table_path, index=False)


if __name__ == "__main__":
    # update_all_table(month='2024-12-01', is_force=False)
    draw_graph()
