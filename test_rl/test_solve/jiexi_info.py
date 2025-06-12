from test_rl.test_script.utils import load_dictionary

info_dict = load_dictionary('/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1110_pre_SMTimer_llama3.1:70b_1200s_info_dict_rl.txt')
lsmod_info = info_dict['/home/lz/baidudisk/smt/buzybox_angr.tar.gz/single_test/lsmod/lsmod1111306']
print(type(lsmod_info))
print(len(lsmod_info[7]))


# print(len(lsmod_info))
count = 0
for i in lsmod_info[7]:
    print(i)
    print(len(i))
    count += len(i)
print(count,count/len(lsmod_info[7]))









