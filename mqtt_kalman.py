import json
import time
from model import Model

# Connect using mqtt
import paho.mqtt.client as mqtt

# User argparser to get username and password for mqtt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--username", help="mqtt username", default="")
parser.add_argument("--password", help="mqtt password", default="")
# Add hostname to the mqtt client
parser.add_argument("--hostname", help="mqtt hostname", default="localhost")
args = parser.parse_args()

# Connect to broker at rollspelswiki.se
broker_address = args.hostname
client = mqtt.Client("kalman")

client.username_pw_set(args.username, args.password)
client.connect(broker_address)

# Listen to topic "sauna/temp1"
client.subscribe("sauna/temp1")

# Create an initial kalman model
md = Model(11.5)

# Keep track of the time
last_time = time.time()


# Define a callback function for sauna temperature, which is a json message where temp2 contains the temperature
def on_message(client, userdata, message):
    global last_time
    # Convert the message to a json object
    message = json.loads(message.payload.decode("utf-8"))
    # Get the temperature
    temp = message["temp2"]
    # Predict the temperature
    seconds_elapsed = time.time() - last_time
    last_time = time.time()
    # Print seconds_elapsed
    print(seconds_elapsed)

    m, p = md.predict(seconds_elapsed)
    # Update the kalman model
    m, p = md.update(temp)
    # Print the predicted and updated temperature
    print("Temp:", m[0], "Rate:", m[1] * 60)

    # Post the updated prediction to "sauna/temp1_prediction"
    client.publish("sauna/temp1_prediction", json.dumps({"temp": m[0], "rate": m[1] * 60}))


# Register the callback function
client.on_message = on_message

# Keep the program running
client.loop_forever()
