from datetime import datetime


def print_timestamp(s):
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y - %H:%M:%S")
    print("{}: {}".format(date_time, s))


print_timestamp("test")