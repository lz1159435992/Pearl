from test_rl.test_script.utils import load_dictionary

file_time = load_dictionary('info_dict_4.4.txt')
time_succeed_1 = 0
time_succeed_2 = 0

time_reduce_1 = 0
time_reduce_2 = 0

count_succeed = 0
count_time = 0
for k,v in file_time.items():
    if v[5] == 'succeed':
        time_succeed_1 += float(v[1])
        time_succeed_2 += float(v[4])
        count_succeed += 1
        if float(v[1]) > float(v[4]):
            time_reduce_1 += float(v[1])
            time_reduce_2 += float(v[4])
            count_time += 1
            print(k)
            print('+++++++++++++++++')
            print(v[1], v[4])
            print('+++++++++++++++++')
        else:
            print('.................')
            print(v[1], v[4])
            print('.................')
print(len(file_time),count_succeed, count_time)
print(time_succeed_1, time_succeed_2, time_reduce_1, time_reduce_2)