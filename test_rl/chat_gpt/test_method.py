def merge_intervals(intervals):
    # 首先根据区间的起始点对区间进行排序
    intervals.sort(key=lambda x: x[0])

    # 初始化合并后的区间列表
    merged = []

    # 遍历排序后的区间列表
    for interval in intervals:
        # 如果合并列表为空，或者当前区间的起始点大于合并列表中最后一个区间的结束点
        if not merged or merged[-1][1] < interval[0]:
            # 直接添加当前区间到合并列表
            merged.append(interval)
        else:
            # 否则，合并当前区间与合并列表中最后一个区间
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged


import random


def calculate_total_numbers(intervals):
    total_numbers = 0
    for start, end in intervals:
        total_numbers += end - start + 1
    return total_numbers


def distribute_intervals_evenly(intervals, m):
    # 计算总的整数数量
    total_numbers = sum(end - start + 1 for start, end in intervals)

    # 如果 m 大于总的整数数量，则直接返回原始的输入列表
    if m > total_numbers:
        return [(start, end) for start, end in intervals]

    # 计算小区间的数量 n
    n = total_numbers // m
    # 如果 m 不能整除总的整数数量，增加一个小区间
    if total_numbers % m != 0:
        n += 1

    # 初始化小区间列表
    subintervals = [[] for _ in range(n)]

    # 遍历原始区间列表，分配整数到每个小区间
    index = 0
    for start, end in intervals:
        while start <= end:
            # 将当前整数分配到当前小区间
            subintervals[index % n].append(start)
            start += 1
            # 更新当前分配的整数数量
            index += 1

    # 将每个小区间的整数列表转换为区间列表
    subintervals_as_ranges = []
    for subinterval in subintervals:
        subintervals_as_ranges.append(list(zip(subinterval, subinterval)))

    return subintervals_as_ranges

#另一种实现方法
# def divide_intervals(intervals, n):
#     # 计算总的整数个数
#     total_count = sum(end - start + 1 for start, end in intervals)
#
#     # 计算每个小区间应该包含的整数个数
#     if n == 0 or total_count == 0:
#         return []
#
#     m = total_count // n
#     if m == 0:
#         return [intervals]
#
#     result = []
#     current_interval = []
#     current_count = 0
#
#     for interval in intervals:
#         start, end = interval
#         length = end - start + 1
#
#         while length > 0:
#             if current_count + length <= m:
#                 current_interval.append([start, end])
#                 current_count += length
#                 break
#             else:
#                 part_length = m - current_count
#                 current_interval.append([start, start + part_length - 1])
#                 result.append(current_interval)
#                 start += part_length
#                 length -= part_length
#                 current_interval = []
#                 current_count = 0
#
#         if current_count == m:
#             result.append(current_interval)
#             current_interval = []
#             current_count = 0
#
#     if current_interval:
#         result.append(current_interval)
#
#     # 平均分配剩余的整数到各小区间
#     remaining = total_count % n
#     if remaining:
#         index = 0
#         for i in range(remaining):
#             result[index].append([start + i, start + i])
#             index = (index + 1) % n
#
#     return result

def divide_intervals(intervals, m):
    # 计算总的整数个数
    total_count = sum(end - start + 1 for start, end in intervals)

    if m >= total_count:
        return [(start, end) for start, end in intervals]

        # 计算小区间的数量 n
    n = total_count // m
    # 如果 m 不能整除总的整数数量，增加一个小区间
    if total_count % m != 0:
        n += 1
    # 计算每个小区间应该包含的整数个数
    if n == 0 or total_count == 0:
        return []

    # m = total_count // n
    # if m == 0:
    #     return [intervals]

    result = []
    current_interval = []
    current_count = 0

    for interval in intervals:
        start, end = interval
        length = end - start + 1

        while length > 0:
            if current_count + length <= m:
                current_interval.append([start, end])
                current_count += length
                break
            else:
                part_length = m - current_count
                current_interval.append([start, start + part_length - 1])
                result.append(current_interval)
                start += part_length
                length -= part_length
                current_interval = []
                current_count = 0

        if current_count == m:
            result.append(current_interval)
            current_interval = []
            current_count = 0

    if current_interval:
        result.append(current_interval)

    # 平均分配剩余的整数到各小区间
    remaining = total_count % n
    if remaining:
        index = 0
        for i in range(remaining):
            result[index].append([start + i, start + i])
            index = (index + 1) % n

    return result


def random_from_subinterval(subintervals, n):
    # 随机选择第n个小区间
    chosen_subinterval = subintervals[n - 1]

    # 计算这个小区间中包含的整数个数
    total_numbers = 0
    for start, end in chosen_subinterval:
        total_numbers += (end - start + 1)

    # 随机选择一个整数索引
    random_index = random.randint(1, total_numbers)

    # 初始化计数器
    count = 0
    for start, end in chosen_subinterval:
        # 如果随机索引在当前区间内
        if count + (end - start + 1) >= random_index:
            # 返回对应的整数
            return start + (random_index - count) - 1
        count += (end - start + 1)  # 更新计数器

def distribute_intervals_by_fixed_count(intervals, m):
    total_numbers = sum(end - start + 1 for start, end in intervals)
    n = total_numbers // m  # 计算需要的小区间数量

    # 如果有剩余的整数，增加一个小区间
    if total_numbers % m != 0:
        n += 1

    # 初始化小区间列表
    subintervals = [[] for _ in range(n)]
    current_subinterval_index = 0

    # 遍历所有整数并分配到小区间
    for start, end in intervals:
        for i in range(start, end + 1):
            subintervals[current_subinterval_index].append(i)
            # 移动到下一个小区间的起始位置
            if len(subintervals[current_subinterval_index]) >= m:
                current_subinterval_index += 1

    # 如果最后一个小区间没有达到m，它将包含剩余的所有整数
    return subintervals
# 示例使用
intervals = [[1, 2], [5, 6], [8, 10], [15, 16], [17, 20], [25, 30],[200,500],[100,700]]
subintervals = merge_intervals(intervals)
print(subintervals)
subintervals = divide_intervals(subintervals,10)
print(subintervals)
# 示例使用
# 假设subintervals已经是根据前面的逻辑分配好的小区间列表
# subintervals = [
#     [[1, 3], [10, 12]],  # 第一个小区间
#     [[4, 4], [5, 9]],  # 第二个小区间
#     [[13, 20]]  # 第三个小区间
# ]
n = 5 # 假设我们想从第二个小区间中随机选择一个整数

# 从第n个小区间中随机选择一个整数
random_number = random_from_subinterval(subintervals, 1)
print(f"Randomly selected number from the {n}th subinterval: {random_number}")

# 示例使用
intervals = [[1, 10], [12, 15], [17, 20]]  # 假设这是已经合并和排序的区间列表
n = 3  # 将整数平均分成3个小区间
variable = 7  # 给定的变量值

# 首先将整数平均分配到n个小区间
# subintervals = d distribute_numbers_evenly(intervals, n))

# # 然后从对应的小区间中随机选择一个整数
# random_number = random_from_subinterval(subintervals, variable)
# print(f"Randomly selected number: {random_number}")
