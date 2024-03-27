from test_rl.test_script.utils import load_dictionary

file_time = load_dictionary('info_dict.txt')
count_succeed = 0
count_time = 0
for k,v in file_time.items():
    if v[5] == 'succeed':
        count_succeed += 1
        if float(v[1]) > float(v[4]):
            count_time += 1
            print(v)
            print(v[1],v[4])
print(len(file_time),count_succeed, count_time)