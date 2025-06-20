# SMT Solver CLI 使用说明

这是一个重构后的SMT求解器命令行工具，支持多种SMT求解器：Z3, CVC5, MathSAT5, OpenSMT, Yices。

## 功能特性

- 支持多种SMT求解器
- 批量求解文件
- 单文件求解
- 并发处理（多线程/多进程）
- 结果保存和恢复
- 路径替换功能
- 超时控制

## 安装依赖

确保已安装以下依赖：
```bash
pip install z3-solver loguru
```

同时确保系统中安装了相应的SMT求解器：
- CVC5: `cvc5`
- MathSAT5: `mathsat`
- OpenSMT: `opensmt`
- Yices: `yices-smt2`

## 使用方法

### 1. 批量求解

```bash
# 基本用法
python smt_solver_cli.py batch --info-file info_dict.txt --solver z3

# 指定超时时间和并发数
python smt_solver_cli.py batch --info-file info_dict.txt --solver cvc5 --timeout 600 --max-workers 8

# 指定输出文件
python smt_solver_cli.py batch --info-file info_dict.txt --solver mathsat5 --output my_results.json

# 路径替换（用于不同环境间的路径映射）
python smt_solver_cli.py batch --info-file info_dict.txt --solver z3 --path-replace /old/path /new/path
```

### 2. 单文件求解

```bash
# 求解单个文件
python smt_solver_cli.py single --file /path/to/file.smt2 --solver z3 --timeout 10
```

### 3. 运行测试

```bash
# 运行基本测试用例
python smt_solver_cli.py test
```

### 4. 查看帮助

```bash
# 查看总体帮助
python smt_solver_cli.py --help

# 查看批量求解帮助
python smt_solver_cli.py batch --help

# 查看单文件求解帮助
python smt_solver_cli.py single --help
```

## 参数说明

### 批量求解参数

- `--info-file`: 包含文件路径信息的文件（必需）
- `--solver`: 求解器名称，可选值：z3, cvc5, mathsat5, opensmt, yices（必需）
- `--timeout`: 超时时间（秒），默认1200
- `--max-workers`: 最大并发数，默认4
- `--output`: 输出结果文件名，默认自动生成
- `--path-replace`: 路径替换，格式为"旧路径 新路径"，可多次使用

### 单文件求解参数

- `--file`: 要求解的文件路径（必需）
- `--solver`: 求解器名称（必需）
- `--timeout`: 超时时间（秒），默认5

## 输入文件格式

### 信息字典文件格式

信息字典文件应该包含文件路径到相关信息的映射，例如：
```
/path/to/file1.smt2
/path/to/file2.smt2
/path/to/file3.smt2
```

### SMT文件格式

支持两种格式：
1. 纯SMT-LIB格式
2. JSON格式，包含`script`或`smt_script`字段

## 输出格式

结果保存为JSON格式，结构如下：
```json
{
  "file_path": [
    "result",           // 求解结果：sat/unsat/unknown/timeout/error
    "solve_time",       // 求解时间（秒）
    "timeout",          // 设置的超时时间
    "model"             // 模型信息（字典或字符串）
  ]
}
```

## 与原脚本的对应关系

| 原函数 | 新命令 | 说明 |
|--------|--------|------|
| `get_Z3_result()` | `batch --solver z3` | Z3批量求解 |
| `get_CVC5_result()` | `batch --solver cvc5` | CVC5批量求解 |
| `get_MathSAT_result()` | `batch --solver mathsat5` | MathSAT5批量求解 |
| `get_OpenSMT_result()` | `batch --solver opensmt` | OpenSMT批量求解 |
| `get_Yices_result()` | `batch --solver yices` | Yices批量求解 |

## 示例

### 示例1：使用Z3求解器批量处理

```bash
python smt_solver_cli.py batch \
  --info-file smtimer-533-result.txt \
  --solver z3 \
  --timeout 1200 \
  --max-workers 8 \
  --path-replace /home/lz/baidudisk/ /home/nju/Downloads/ \
  --output z3_results.json
```

### 示例2：使用CVC5求解单个文件

```bash
python smt_solver_cli.py single \
  --file /path/to/test.smt2 \
  --solver cvc5 \
  --timeout 30
```

### 示例3：使用MathSAT5进行快速测试

```bash
python smt_solver_cli.py batch \
  --info-file test_files.txt \
  --solver mathsat5 \
  --timeout 60 \
  --max-workers 4
```

## 注意事项

1. 确保求解器已正确安装并可在PATH中找到
2. 对于大文件集，建议使用适当的并发数以避免系统过载
3. 结果文件会自动保存，可以中断后重新运行以继续处理
4. Z3求解器使用多进程，其他求解器使用多线程
5. 路径替换功能对于在不同环境间迁移很有用

## 故障排除

### 常见问题

1. **求解器未找到**: 确保求解器已安装并在PATH中
2. **权限错误**: 确保有读取输入文件和写入输出文件的权限
3. **内存不足**: 减少并发数或超时时间
4. **路径问题**: 使用`--path-replace`参数处理路径差异

### 调试模式

可以通过修改日志级别来获取更详细的调试信息：
```python
# 在脚本开头添加
logger.add(sys.stderr, level="DEBUG")
``` 