# /usr/bin/python
import arrow

def get_months(start_month, end_month):
    """
    :param start_month: month in YYY-MM
    :param end_month: month in YYY-MM
    :return: list of months in text format
    """
    # we convert to arrows:
    start = arrow.get(start_month + "-01")
    end = arrow.get(end_month + "-01")
    periods = []
    current = start
    while current <= end:
        periods.append(current.format("YYYY-MM"))
        current = current.shift(months=1)
    return periods


def get_prev_month(month):
    return arrow.get(month+ "-01").shift(months=-1).format("YYYY-MM")