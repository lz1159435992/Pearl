import json

with open('result_dict_no_increment.txt', 'r') as file:
    result_dict = json.load(file)
for k,v in result_dict.items():
    if v[0] == 'sat' and v[1] > 100:
        print(k, v)

