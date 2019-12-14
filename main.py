from dl_1 import lab as lab1
from dl_2 import lab as lab2
from dl_3 import lab as lab3
from dl_4 import lab as lab4
from dl_5 import lab as lab5
from dl_6 import lab as lab6
from dl_7 import lab as lab7
from dl_8 import lab as lab8

DEFAULT_RUNNABLE_LAB = "1"


def main():
    if DEFAULT_RUNNABLE_LAB == "-1":
        print("Which ML lab do you want to start?")
        user_input = input()
    else:
        user_input = DEFAULT_RUNNABLE_LAB
    options = {
        "1": lab1.Lab1(),
        "2": lab2.Lab2(),
        "3": lab3.Lab3(),
        "4": lab4.Lab4(),
        "5": lab5.Lab5(),
        "6": lab6.Lab6(),
        "7": lab7.Lab7(),
        "8": lab8.Lab8(),
    }
    options[user_input].run()


if __name__ == '__main__':
    main()
