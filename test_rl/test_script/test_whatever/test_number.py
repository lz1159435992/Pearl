from decimal import Decimal, getcontext

# 设置decimal的精度，这里设置为40位，可以根据需要调整
# getcontext().prec = 40

# number = Decimal('3.79162643983031241231348895e+18')
number = 564354444444444444444444415646543654654654564682313555554545246546345655554444423444
# 转换为字符串，不使用科学计数法
# formatted_number = int(format(number, 'f'))
formatted_number = str(number)
print(formatted_number)