import tdmclient.notebook
@tdmclient.notebook.sync_var


def printing():
    print('cc')

def button_right():
    global motor_left_target, motor_right_target
    print('salut')
    motor_left_target = 100
    motor_right_target = -100

while(True):
    printing()
    button_right()
