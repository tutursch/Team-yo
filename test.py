import tdmclient.notebook


def printing():
    print('Soif')

@onevent 
def button_right():
    global motor_left_target, motor_right_target
    print('salut')
    motor_left_target = 100
    motor_right_target = -100