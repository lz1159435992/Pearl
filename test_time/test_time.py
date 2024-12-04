import signal
import time
####################################
class MyException(Exception):
    def __init__(self, value):
        self.value = value
def timeout_handler(signum, frame):
    print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    raise MyException("Timeout!")
####################################
def a_long_cost_time_func():
    cnt=0
    while True:
        cnt+=1
        print(f"Hello-{cnt}")
        time.sleep(0.5)
####################################
signal.alarm(6)
signal.signal(signal.SIGALRM, timeout_handler)
try:
    a_long_cost_time_func()
except MyException as e:
    print("Time out! Bye!")
signal.alarm(0)