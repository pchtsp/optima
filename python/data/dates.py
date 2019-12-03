# /usr/bin/python3

import arrow
import datetime

def get_months(start_month, end_month):
    """
    :param start_month: month in YYY-MM
    :param end_month: month in YYY-MM
    :return: list of months in text format
    """
    # we convert to arrows:
    start = arrow.get(start_month + "-01")
    end = arrow.get(end_month + "-01")
    return [date.format("YYYY-MM") for date in arrow.Arrow.range('month', start, end)]


def shift_month(month, shift=1):
    return arrow.get(month + "-01").shift(months=shift).format("YYYY-MM")


def month_to_arrow(month):
    return arrow.get(month + "-01")


def get_prev_month(month):
    return shift_month(month, shift=-1)


def get_next_month(month):
    return shift_month(month, shift=1)


def get_timestamp(form="%Y%m%d%H%M"):
    return datetime.datetime.now().strftime(form)


if __name__ == "__main__":
    pass