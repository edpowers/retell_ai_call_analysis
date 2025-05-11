import datetime

import pytz


def get_timestamp_ms(hours_ago: int) -> int:
    # Get timestamp for 24 hours ago in milliseconds
    return (
        int(
            (
                datetime.datetime.now(tz=pytz.timezone("US/Eastern"))
                - datetime.timedelta(hours=hours_ago)
            ).timestamp()
        )
        * 1000
    )
