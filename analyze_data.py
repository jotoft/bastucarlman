# %%
from datetime import datetime
import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import matplotlib.pyplot as plt

# Add argparser for token org and bucket
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--token", help="influxdb token", default="")
parser.add_argument("--org", help="influxdb org", default="")
parser.add_argument("--bucket", help="influxdb bucket", default="")
args = parser.parse_args()

# You can generate a Token from the "Tokens Tab" in the UI
token = args.token
org = args.org
bucket = args.bucket

client = InfluxDBClient(url="http://rollspelswiki.se:8086", token=token)

query = rf"""from(bucket: "{bucket}")
        |> range(start:-8h, stop: -4h)
        |> filter(fn: (r) =>
             r._measurement == "tempsensor" and
             r._field == "temp" and
            r.device == "cubecell"
        )"""
df = client.query_api().query_data_frame(query, org=org)
# %%
df['_value'].plot()
plt.show()

# %%
time = df['_time']

# Get delta times in seconds

# %%
test = time[1] - time[0]

delta_times = [(t2 - t1).delta / 1E9 for t1, t2 in zip(time, time[1:])]

print(delta_times)

# %%
from model import Model

md = Model(11.5)

# Create new dataframe to collect kalman filtered values
df_kalman = pd.DataFrame(columns=['temp', 'rate'])

for dt, value in zip(delta_times, df['_value'][1:]):
    m_, p_ = md.predict(dt)

    # print(m_[0], m_[1])
    m, p = md.update(value)
    print(m[0], m[1] * 60.0)

    df_kalman.loc[len(df_kalman)] = [m[0], m[1] * 60.0]

# Use index from df_kalman to match with df
df_kalman.index = df.index[1:]

# Use the timestamps from df to index, they are in df['_time']
df_kalman.index = df['_time'][1:]

# %%
# Plot the kalman filtered values
# Plot rate in separate axis
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# Plot the kalman filtered values
ax1.plot(df_kalman['temp'], color='red')
ax2.plot(df_kalman['rate'], color='blue')

# Fix the messy axis labels
ax1.set_ylabel('Temperature [°C]', color='red')
ax2.set_ylabel('Rate [°C/min]', color='blue')

ax2.set_xlabel('Time')

# Show the plot
plt.show()
