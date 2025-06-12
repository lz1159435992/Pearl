import asyncio
from pyecharts_snapshot.main import make_a_snapshot

# 异步函数保存为 PNG 或 PDF
async def save_snapshot(input_file, output_file):
    await make_a_snapshot(input_file, output_file)

# 主函数
async def main():
    input_file = 'Z3solver/buzybox_counts.html'
    output_file = 'Z3solver/buzybox_counts.png'
    await save_snapshot(input_file, output_file)

# 运行主函数
if __name__ == "__main__":
    asyncio.run(main())