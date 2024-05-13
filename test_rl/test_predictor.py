import json
import pickle

from pearl.SMTimer.KNN_Predictor import Predictor
from z3 import *

from test_rl.test_script.utils import solve_and_measure_time, load_dictionary


def test_pre():
    predictor = Predictor('KNN')
    with open('smt4.18.json', 'r') as file:
        strings = json.load(file)
    timeout = 99999999
    result_dict = {}
    for k, s in strings.items():
        result_list = []
        assertions = parse_smt2_string(s)
        solver = Solver()
        for a in assertions:
            solver.add(a)
        result, model, time_taken = solve_and_measure_time(solver, timeout)
        result_list.append(result)
        result_list.append(time_taken)
        result_list.append(timeout)
        predict = predictor.predict(s)

        actual = result  # 实际结果
        actual_time = time_taken  # 实际消耗的时间
        prediction = predict  # 模型的预测结果

        # 根据消耗的时间调整预测结果
        if actual_time > 200:
            prediction = 'unsat' if prediction == 'sat' else prediction

        # 更新统计量
        if prediction == 'sat':
            if actual == 'sat':
                predictor.increment_KNN_data(0)
            else:
                predictor.increment_KNN_data(1)
        else:  # prediction == 'unsat'
            if actual == 'unsat':
                predictor.increment_KNN_data(1)
            else:
                predictor.increment_KNN_data(0)

        result_list.append(int(predict))
        result_dict[k] = result_list
        with open('pre_result.txt', 'w') as file:
            json.dump(result_dict, file, indent=4)

    with open('predictor.pkl', 'wb') as f:
        pickle.dump(predictor, f)

        for k, s in strings.items():
            predict = predictor.predict(s)
            result_dict[k].append(int(predict))

        with open('pre_result_after.txt', 'w') as file:
            json.dump(result_dict, file, indent=4)


def get_test():
    with open('pre_result_after.txt', 'r') as file:
        # 璇诲彇鏂囦欢鎵€鏈夊唴瀹瑰埌涓€涓瓧绗︿覆
        pre_str = file.read()
        # 灏咼SON瀛楃涓茶浆鎹负瀛楀吀
        dict_obj = json.loads(pre_str)
    print(dict_obj)
    #未训练前
    count_tp = 0  # True Positives
    count_fp = 0  # False Positives
    count_tn = 0  # True Negatives
    count_fn = 0  # False Negatives

    # 遍历字典，计算统计量
    for k, v in dict_obj.items():
        actual = v[0]  # 实际结果
        actual_time = v[1]  # 实际消耗的时间
        prediction = v[3]  # 模型的预测结果

        # 根据消耗的时间调整预测结果
        if actual_time > 200:
            actual = 'unsat' if actual == 'sat' else actual

        # 更新统计量
        if prediction == 0:
            if actual == 'sat':
                count_tp += 1  # 正确预测'sat'
            else:
                count_fp += 1  # 错误预测'sat'
        else:  # prediction == 'unsat'
            if actual == 'unsat':
                count_tn += 1  # 正确预测'unsat'
            else:
                count_fn += 1  # 错误预测'unsat'

    # 计算准确率、召回率和准确率
    total_predictions = len(dict_obj)
    accuracy = (count_tp + count_tn) / total_predictions if total_predictions > 0 else 0
    precision = count_tp / (count_tp + count_fp) if (count_tp + count_fp) > 0 else 0
    recall = count_tp / (count_tp + count_fn) if (count_tp + count_fn) > 0 else 0

    # 打印结果
    print(f"Count TP: {count_tp}, Count FP: {count_fp}, Count TN: {count_tn}, Count FN: {count_fn}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Total entries in dict_obj: {total_predictions}")
    #训练后
    count_tp = 0  # True Positives
    count_fp = 0  # False Positives
    count_tn = 0  # True Negatives
    count_fn = 0  # False Negatives

    # 遍历字典，计算统计量
    for k, v in dict_obj.items():
        actual = v[0]  # 实际结果
        actual_time = v[1]  # 实际消耗的时间
        prediction = v[4]  # 模型的预测结果

        # 根据消耗的时间调整预测结果
        if actual_time > 200:
            actual = 'unsat' if actual == 'sat' else actual

        # 更新统计量
        if prediction == 0:
            if actual == 'sat':
                count_tp += 1  # 正确预测'sat'
            else:
                count_fp += 1  # 错误预测'sat'
        else:  # prediction == 'unsat'
            if actual == 'unsat':
                count_tn += 1  # 正确预测'unsat'
            else:
                count_fn += 1  # 错误预测'unsat'

    # 计算准确率、召回率和准确率
    total_predictions = len(dict_obj)
    accuracy = (count_tp + count_tn) / total_predictions if total_predictions > 0 else 0
    precision = count_tp / (count_tp + count_fp) if (count_tp + count_fp) > 0 else 0
    recall = count_tp / (count_tp + count_fn) if (count_tp + count_fn) > 0 else 0

    # 打印结果
    print(f"Count TP: {count_tp}, Count FP: {count_fp}, Count TN: {count_tn}, Count FN: {count_fn}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Total entries in dict_obj: {total_predictions}")


def test_pre_2():
    predictor = Predictor('KNN')
    with open('smt.json', 'r') as file:
        strings = json.load(file)

    seen = set()
    unique_list = [x for x in strings if not (x in seen or seen.add(x))]
    print(unique_list)

    timeout = 99999999
    result_dict = {}
    count = 0
    for s in unique_list:
        result_list = []
        assertions = parse_smt2_string(s)
        solver = Solver()
        for a in assertions:
            solver.add(a)
        result, model, time_taken = solve_and_measure_time(solver, timeout)
        result_list.append(result)
        result_list.append(time_taken)
        result_list.append(timeout)
        predict = predictor.predict(s)
        result_list.append(int(predict))
        result_dict[count] = result_list
        count += 1
        # with open('pre_result.txt', 'w') as file:
        #     json.dump(result_dict, file, indent=4)


def change_strings():
    with open('smt.json', 'r') as file:
        strings = json.load(file)

    seen = set()
    unique_list = [x for x in strings if not (x in seen or seen.add(x))]
    print(unique_list)
    dict_obj = {index: string for index, string in enumerate(unique_list)}

    # 将字典转换为JSON字符串
    json_str = json.dumps(dict_obj, indent=4)

    # 打印JSON字符串查看结果
    print(json_str)

    # 如果需要将JSON字符串保存到文件
    with open('smt4.18.json', 'w') as json_file:
        json.dump(dict_obj, json_file, indent=4)

def test_pre_after():

    # with open('smt.json', 'r') as file:
    #     strings = json.load(file)
    # print(len(strings))
    # get_test()
    # test_pre()
    result_dict = load_dictionary('pre_result.txt')
    predictor = Predictor('KNN')
    # with open('pre_result.txt', 'w') as file:
    #     json.dump(result_dict, file, indent=4)

    # with open('predictor.pkl', 'rb') as f:
    #     predictor = pickle.load(f)
    with open('smt4.18.json', 'r') as file:
        strings = json.load(file)
        print(strings)
        for k, s in strings.items():
            # result_list = []
            # assertions = parse_smt2_string(s)
            # solver = Solver()
            # for a in assertions:
            #     solver.add(a)
            # result, model, time_taken = solve_and_measure_time(solver, timeout)
            # result_list.append(result)
            # result_list.append(time_taken)
            # result_list.append(timeout)
            predict = predictor.predict(s)

            actual = result_dict[k][0]  # 实际结果
            actual_time = result_dict[k][1]  # 实际消耗的时间
            prediction = predict  # 模型的预测结果

            # 根据消耗的时间调整预测结果
            if actual_time > 200:
                prediction = 'unsat' if prediction == 'sat' else prediction

            # 更新统计量
            if prediction == 'sat':
                if actual == 'sat':
                    predictor.increment_KNN_data(0)
                else:
                    predictor.increment_KNN_data(1)
            else:  # prediction == 'unsat'
                if actual == 'unsat':
                    predictor.increment_KNN_data(1)
                else:
                    predictor.increment_KNN_data(0)

        for k, s in strings.items():
            predict = predictor.predict(s)
            result_dict[k].append(int(predict))

        with open('pre_result_after.txt', 'w') as file:
            json.dump(result_dict, file, indent=4)

if __name__ == '__main__':

    # with open('pre_result_after.txt', 'w') as file:
    #     json.dump(result_dict, file, indent=4)
    get_test()