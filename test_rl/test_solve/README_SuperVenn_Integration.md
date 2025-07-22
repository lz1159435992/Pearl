# SuperVenn 集成功能使用说明

## 概述

本文档说明如何在 `test_group_cvc5_process_analysis` 方法中使用新增的 SuperVenn 绘制功能。该功能仿照 `/home/lz/PycharmProjects/Pearl/test_rl/AriParti_sync/scripts/compare_results.py` 中的方法实现。

## 功能特性

- **自动生成 SuperVenn 图**: 比较基线求解器和 RL+LLM 增强版本的求解能力
- **专业可视化**: 使用学术标准的颜色方案和字体设置
- **灵活输出**: 支持指定输出目录或跳过图片生成
- **详细统计**: 提供完整的求解能力分析统计信息

## 使用方法

### 基本用法

```python
from test_rl.test_solve.test_solver_result import test_group_cvc5_process_analysis

# 基本调用（不生成SuperVenn图）
result_dict, time_dict, time_dict_2, info_dict, supervenn_stats = test_group_cvc5_process_analysis(
    info_name='/path/to/info_dict_file.txt',
    var_count_path='/path/to/var_count.txt'
)
```

### 生成 SuperVenn 图

```python
# 指定输出目录生成SuperVenn图
output_dir = '/path/to/output/directory'
result_dict, time_dict, time_dict_2, info_dict, supervenn_stats = test_group_cvc5_process_analysis(
    info_name='/path/to/info_dict_file.txt',
    var_count_path='/path/to/var_count.txt',
    output_dir=output_dir,
    solver_name="CVC5"  # 可自定义求解器名称
)
```

### 完整示例

```python
import os
from test_rl.test_solve.test_solver_result import test_group_cvc5_process_analysis

# 设置文件路径
info_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/cvc5_process/info_dict_SMTimer_llama3.1:70b_1200s_info_dict_rl_cvc5_0628.txt'
var_count_path = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/cvc5_smtimer_var_count.txt'
output_dir = '/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/supervenn_output'

# 执行分析并生成SuperVenn图
result_dict, time_dict, time_dict_2, info_dict, supervenn_stats = test_group_cvc5_process_analysis(
    info_name=info_name,
    var_count_path=var_count_path,
    output_dir=output_dir,
    solver_name="CVC5"
)

# 查看SuperVenn统计信息
if supervenn_stats:
    print("SuperVenn 分析结果:")
    print(f"  基线求解器独有: {supervenn_stats['baseline_total'] - len(supervenn_stats['both_solved'])} 个约束")
    print(f"  RL+LLM 独有: {supervenn_stats['rl_llm_total'] - len(supervenn_stats['both_solved'])} 个约束")
    print(f"  两者都解决: {len(supervenn_stats['both_solved'])} 个约束")
    print(f"  总计解决: {supervenn_stats['union_total']} 个约束")
```

## 参数说明

### 函数参数

- **info_name** (str): CVC5处理结果文件路径
- **var_count_path** (str): 变量统计文件路径，默认为 `'/home/lz/PycharmProjects/Pearl/test_rl/test_solve/var_count.txt'`
- **output_dir** (str, optional): SuperVenn图输出目录，如果为 `None` 则不生成图片
- **solver_name** (str): 求解器名称，用于图表标签，默认为 `"CVC5"`

### 返回值

函数返回一个包含5个元素的元组：

1. **result_dict**: 分类结果字典
2. **time_dict**: 原始求解器时间字典
3. **time_dict_2**: RL+LLM时间字典
4. **info_dict**: 处理后的完整信息字典
5. **supervenn_stats**: SuperVenn统计信息字典（如果未生成图片则为 `None`）

### SuperVenn 统计信息结构

```python
supervenn_stats = {
    'baseline_only': set(),      # 仅基线求解器解决的约束集合
    'rl_llm_only': set(),        # 仅RL+LLM解决的约束集合
    'both_solved': set(),        # 两者都解决的约束集合
    'baseline_total': int,       # 基线求解器解决的总数
    'rl_llm_total': int,         # RL+LLM解决的总数
    'union_total': int           # 两者合计解决的总数
}
```

## 输出文件

当指定 `output_dir` 时，函数会生成以下文件：

- **supervenn_{solver_name.lower()}_comparison.pdf**: SuperVenn比较图（PDF格式，300 DPI）

例如，当 `solver_name="CVC5"` 时，生成的文件名为 `supervenn_cvc5_comparison.pdf`。

## 依赖要求

确保已安装以下Python包：

```bash
pip install supervenn matplotlib
```

## 测试

可以运行测试脚本验证功能：

```bash
python test_rl/test_solve/test_supervenn_integration.py
```

## 注意事项

1. **文件路径**: 确保输入文件路径正确且文件存在
2. **输出目录**: 如果输出目录不存在，函数会自动创建
3. **数据格式**: 确保输入数据格式符合预期的CVC5处理结果格式
4. **内存使用**: 处理大量数据时注意内存使用情况

## 错误处理

函数包含完善的错误处理机制：

- 自动检查数据完整性
- 处理缺失的变量统计信息
- 优雅处理SuperVenn图生成失败的情况
- 提供详细的错误信息和警告

## 扩展性

该实现具有良好的扩展性，可以轻松适配其他求解器：

- 修改 `solver_name` 参数
- 调整数据结构解析逻辑
- 自定义颜色方案和图表样式

## 示例输出

成功运行后，控制台会显示类似以下的输出：

```
Total Count: 555, Succeed Count: 357, Failed Count: 198
...
Generating SuperVenn diagrams...
Created directory: /path/to/output
SuperVenn diagram saved to: /path/to/output/supervenn_cvc5_comparison.pdf

CVC5 SuperVenn Analysis:
  CVC5 Baseline only solved: 73 constraints
  CVC5 + RL+LLM only solved: 317 constraints
  Both methods solved: 40 constraints
  Total unique constraints solved: 430 constraints
```
