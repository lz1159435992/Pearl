import math
def counter_reward_function(total_length, unique_count):

        """
        Calculate the reward based on the total length of the list and the number of unique in it.

        Args:
        - total_length (int): The total length of the list.
        - unique_count (int): The number of unique in the list.

        Returns:
        - float: The calculated reward.
        """
        # Define the base reward values
        R_positive = 1
        R_negative = -1

        # Define the scaling factor for negative reward
        alpha = 1 / math.sqrt(total_length) if total_length > 0 else 1

        # Check if there are any unique strings
        if unique_count > 0:
            # Calculate the positive reward, scaled based on the list length
            reward = R_positive / math.log(1 + total_length)
        else:
            # Apply the negative reward, scaled by alpha
            reward = R_negative * alpha

        return reward

counterexamples_list = [
    # 第1个新子列表
    [
        ['unconstrained_ret_mbrtowc_3174_64', 4272629462947298423]
    ],
    # 第3个新子列表
    [
        ['unconstrained_ret_mbrtowc_4911_64', 1234981982398123981],
        ['unconstrained_ret_mbrtowc_4737_64', 8932941238423795823]
    ],
    # 第4个新子列表
    [
        ['unconstrained_ret_mbrtowc_2647_64', 9209188336860664896],
        ['unconstrained_ret_mbrtowc_3874_64', 9072474934692564441],
        ['unconstrained_ret_mbrtowc_3525_64', 9223372036854775807]
    ],
    # 第5个新子列表
    [
        ['unconstrained_ret_mbrtowc_2998_64', 8232947298423795823],
        ['unconstrained_ret_mbrtowc_4558_64', 1234981239812398123],
        ['unconstrained_ret_mbrtowc_2301_64', 8932947298423795823]
    ]
    # # 第2个新子列表（与第1个相同）
    # [
    #     ['unconstrained_ret_mbrtowc_3174_64', 4272629462947298423],
    #     ['unconstrained_ret_mbrtowc_4389_64', 8239482394823795823]
    # ]
]
reward = 0
count = 0
if len(counterexamples_list) > 1:
    if counterexamples_list[-1] in counterexamples_list[:len(counterexamples_list) - 1]:
        reward += -1
    else:
        # 判断新的序列和之前是否有重复（字符串重复）
        # for i in range(len(counterexamples_list) - 1):
        #     # if self.are_lists_equal(counterexamples_list[i],counterexamples_list[-1]):
        #     if ' '.join(counterexamples_list[-1]) in ' '.join(counterexamples_list[i]):
        #         count += 1
        last_joined = ' '.join(
            ' '.join(str(item) for item in inner_list) for inner_list in counterexamples_list[-1])
        for i in range(len(counterexamples_list) - 1):
            current_joined = ' '.join(
                ' '.join(str(item) for item in inner_list) for inner_list in counterexamples_list[i])
            if last_joined in current_joined:
                count += 1
        reward += counter_reward_function(len(counterexamples_list) - 1,
                                               len(counterexamples_list) - 1 - count)
    print(counterexamples_list)
    print(reward)
