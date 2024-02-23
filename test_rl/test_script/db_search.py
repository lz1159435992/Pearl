import sqlite3
import json

def fetch_data_as_dict(db_path, table_name):
    # 连接到SQLite数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 查询表中的所有键值对
    query = f"SELECT key, value FROM {table_name}"
    cursor.execute(query)
    rows = cursor.fetchall()

    # 关闭数据库连接
    conn.close()

    # 将查询结果转换为字典
    result_dict = {row[0]: row[1] for row in rows}
    return result_dict
if __name__ == '__main__':

    db_path = 'result_dictionary.db'
    table_name = 'result_dictionary'
    result_dict = fetch_data_as_dict(db_path, table_name)
    for key, value in result_dict.items():
        list1 = json.loads(value)
        if list1[0] == "unknown":
            print(key)
# conn = sqlite3.connect(db_path)

# 示例数据
# cursor = conn.cursor()
# cursor.execute('SELECT * FROM result_dictionary')
# rows = cursor.fetchall()
# for i in rows:
#     list1 = json.loads(i[1])
#     if list1[0] == "unknown":
#         print(i)
# conn.close()