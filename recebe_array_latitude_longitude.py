from flask import Flask, request, jsonify
import numpy as np
from scipy.spatial.transform import Rotation as R

# Create a Flask app
app = Flask(__name__)

# Define the sensor fusion function
def fuse_sensors(sensor_data):
  """Fuses sensor data from multiple sensors.

  Args:
    sensor_data: A list of sensor data, where each element is a dictionary
      containing the sensor data.

  Returns:
    A fused sensor data dictionary.
  """

  # Get the sensor readings
  sensor_readings = [np.array(sensor["reading"]) for sensor in sensor_data]

  # Fuse the sensor readings
  fused_sensor_reading = np.mean(sensor_readings, axis=0)

  # Return the fused sensor data
  return {
    "fused_reading": fused_sensor_reading.tolist()
  }

# Define the API endpoint
@app.route("/fuse", methods=["POST"])
def fuse():
  """Fuses sensor data from multiple sensors.

  Returns:
    A JSON object containing the fused sensor data.
  """

  # Get the sensor data from the request body
  sensor_data = request.get_json()

  # Fuse the sensor data
  fused_sensor_data = fuse_sensors(sensor_data)

  # Return the fused sensor data as JSON
  return jsonify(fused_sensor_data)

# Define a function to fuse latitude and longitude data
def fuse_latitude_longitude(sensor_data):
  """Fuses latitude and longitude data from multiple sensors.

  Args:
    sensor_data: A list of sensor data, where each element is a dictionary
      containing the sensor data.

  Returns:
    A fused sensor data dictionary containing the latitude and longitude.
  """

  # Get the latitude and longitude readings
  latitude_readings = [sensor["reading"][0] for sensor in sensor_data]
  longitude_readings = [sensor["reading"][1] for sensor in sensor_data]

  # Fuse the latitude and longitude readings
  fused_latitude = np.mean(latitude_readings, axis=0)
  fused_longitude = np.mean(longitude_readings, axis=0)

  # Return the fused sensor data
  return {
    "fused_latitude": fused_latitude,
    "fused_longitude": fused_longitude
  }

# Define an API endpoint to fuse latitude and longitude data
@app.route("/fuse/latitude_longitude", methods=["POST"])
def fuse_latitude_longitude_api():
  """Fuses latitude and longitude data from multiple sensors.

  Returns:
    A JSON object containing the fused latitude and longitude.
  """

  # Get the sensor data from the request body
  sensor_data = request.get_json()

  # Fuse the latitude and longitude data
  fused_sensor_data = fuse_latitude_longitude(sensor_data)

  # Return the fused sensor data as JSON
  return jsonify(fused_sensor_data)

# Start the Flask app
if __name__ == "__main__":
  app.run(debug=True)
