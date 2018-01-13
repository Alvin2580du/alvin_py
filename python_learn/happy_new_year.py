from datetime import datetime
import pprint

pp = pprint.PrettyPrinter()


def Happy_New_year():
    now = datetime.now()
    this_year = now.year
    day = 1
    while this_year == 2018:
        pp.pprint("2018年第{}天: 开开心心，事事顺利！ ".format(day))
        day += 1
        if day > 365:
            exit(1)

if __name__ == "__main__":
    Happy_New_year()
