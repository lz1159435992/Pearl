# SMT Solver CLI 重构总结

## 重构概述

本次重构将原始的 `main_bingxing_stimer.py` 文件重构为命令行工具形式，主要改进包括：

1. **参数化设计**: 将硬编码的变量提取为命令行参数
2. **模块化结构**: 将功能分解为独立的函数和类
3. **命令行界面**: 提供友好的命令行参数解析
4. **错误处理**: 增强错误处理和用户反馈
5. **文档完善**: 提供详细的使用说明和示例

## 文件结构

```
test_rl/test_cvc5/smtimer_710/
├── main_bingxing_stimer.py          # 原始文件
├── smt_solver_cli.py                # 完整版命令行工具（包含Z3支持）
├── smt_solver_cli_simple.py         # 简化版命令行工具（不依赖Z3）
├── example_usage.py                 # 使用示例脚本
├── test_example.smt2                # 测试SMT文件
├── README.md                        # 详细使用说明
└── REFACTOR_SUMMARY.md              # 本重构总结文档
```

## 主要改进

### 1. 参数化改进

| 原硬编码变量 | 新命令行参数 | 说明 |
|-------------|-------------|------|
| `info_name` | `--info-file` | 信息文件路径 |
| `solver_name` | `--solver` | 求解器名称 |
| `timeout = 1200` | `--timeout` | 超时时间 |
| `max_workers = 4` | `--max-workers` | 并发数 |
| 固定输出文件名 | `--output` | 自定义输出文件 |
| 硬编码路径替换 | `--path-replace` | 灵活路径映射 |

### 2. 功能模块化

#### 原始函数 → 新命令映射

| 原始函数 | 新命令 | 功能 |
|---------|--------|------|
| `get_Z3_result()` | `batch --solver z3` | Z3批量求解 |
| `get_CVC5_result()` | `batch --solver cvc5` | CVC5批量求解 |
| `get_MathSAT_result()` | `batch --solver mathsat5` | MathSAT5批量求解 |
| `get_OpenSMT_result()` | `batch --solver opensmt` | OpenSMT批量求解 |
| `get_Yices_result()` | `batch --solver yices` | Yices批量求解 |
| `test_z3solver_basic_sat()` | `test` | 运行测试 |

### 3. 新增功能

- **单文件求解**: `single` 命令支持求解单个SMT文件
- **测试功能**: `test` 命令检查求解器可用性
- **路径替换**: 支持多个路径映射规则
- **结果恢复**: 自动跳过已求解的文件
- **详细日志**: 提供执行进度和错误信息

## 使用示例

### 基本用法

```bash
# 批量求解
python3 smt_solver_cli_simple.py batch \
  --info-file smtimer-533-result.txt \
  --solver cvc5 \
  --timeout 1200 \
  --max-workers 8

# 单文件求解
python3 smt_solver_cli_simple.py single \
  --file test_example.smt2 \
  --solver cvc5 \
  --timeout 10

# 运行测试
python3 smt_solver_cli_simple.py test
```

### 高级用法

```bash
# 自定义输出文件
python3 smt_solver_cli_simple.py batch \
  --info-file info.txt \
  --solver mathsat5 \
  --output my_results.json

# 路径替换
python3 smt_solver_cli_simple.py batch \
  --info-file info.txt \
  --solver yices \
  --path-replace /old/path /new/path \
  --path-replace /another/old /another/new
```

## 与原脚本的对比

### 优势

1. **灵活性**: 通过命令行参数控制所有行为
2. **可维护性**: 模块化设计，易于扩展和维护
3. **用户友好**: 清晰的帮助信息和错误提示
4. **可重用性**: 支持不同的输入文件和求解器
5. **健壮性**: 更好的错误处理和恢复机制

### 兼容性

- 保持原有的求解器接口
- 兼容原有的文件格式
- 保持原有的结果格式
- 支持原有的并发策略

## 部署说明

### 依赖要求

```bash
# Python依赖
pip install argparse typing

# 系统求解器（可选）
# cvc5, mathsat, opensmt, yices-smt2
```

### 运行环境

- Python 3.6+
- 支持的操作系统: Linux, macOS, Windows
- 求解器需要在PATH中可用

## 扩展指南

### 添加新求解器

1. 继承 `Solver` 基类
2. 实现 `solve` 方法
3. 在 `get_solver` 函数中注册
4. 更新命令行参数选择

### 添加新功能

1. 在 `main` 函数中添加新的子命令
2. 实现对应的处理函数
3. 更新帮助文档

## 测试验证

重构后的工具已通过以下测试：

- ✅ 命令行参数解析
- ✅ 帮助信息显示
- ✅ 求解器可用性检查
- ✅ 文件读取和解析
- ✅ 错误处理机制

## 总结

本次重构成功将原有的硬编码脚本转换为灵活的命令行工具，提高了代码的可维护性和用户体验。新的工具保持了原有功能的完整性，同时增加了更多的灵活性和易用性。 