import torch
one_hot_tensors = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0.], device='cuda:0')

print(torch.argmax(one_hot_tensors).item())
variable = {'unconstrained_ret_mbrtowc_2476_64', 'unconstrained_ret_mbrtowc_2998_64', 'unconstrained_ret_mbrtowc_3349_64', 'unconstrained_ret_mbrtowc_3694_64', 'unconstrained_ret_mbrtowc_4219_64', 'mem_2_280_1008', 'unconstrained_ret_mbrtowc_3874_64', 'unconstrained_ret_mbrtowc_4046_64', 'unconstrained_ret_mbrtowc_3174_64', 'unconstrained_ret_mbrtowc_3525_64', 'mem_1_263_8', 'unconstrained_ret_mbrtowc_2647_64', 'strlen_414_64', 'unconstrained_ret_mbrtowc_4389_64', 'unconstrained_ret_mbrtowc_2301_64', 'unconstrained_ret_mbrtowc_2818_64'}
variable_pred = variable[int(torch.argmax(one_hot_tensors).item())]