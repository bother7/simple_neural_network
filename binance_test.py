
# Remember to update the script for the new data when you change this URL
URL = "./bitcoin_2018_5min.csv"

# Uncomment this call when using matplotlib to generate images
# rather than displaying interactive UI.
#import matplotlib
#matplotlib.use('Agg')

from binance.client import Client
import numpy as np
from datetime import datetime
import dateparser
import pytz
import json
from binance.helpers import date_to_milliseconds

# =====================================================================



# =====================================================================
start = "1 Dec, 2017"
end = "1 Mar, 2018"
symbol = "BNBBTC"
interval = Client.KLINE_INTERVAL_1MINUTE

if __name__ == '__main__':


    client = Client("iPImn0wZ0QfRrMF1oVdHZts2KljM446S4l8K5rhpmT3Ja93d2ZtZeBviCRLO2ZXR",
    "g4U49CoSQGLJk5hKo0gGtfjXtLBwrGQqO81tVp3vKPBzaYllBmWvwZvfbJH0xAvB")

    klines = client.get_historical_klines(symbol, interval, "1 day ago UTC")
    print(len(klines))
    with open(
        "Binance_{}_{}_{}-{}.json".format(
            symbol,
            interval,
            date_to_milliseconds(start),
            date_to_milliseconds(end)
        ),
        'w'  # set file write mode
    ) as f:
        f.write(json.dumps(klines))
    #np.savetxt('data.csv', klines, delimiter=',')
