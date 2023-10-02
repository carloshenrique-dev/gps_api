from flask import Flask, request, jsonify
import numpy as np
from filterpy.kalman import KalmanFilter

# Create a Flask app
app = Flask(__name__)

class SensorFusionKalmanFilter:
    def __init__(self, state_dimension, measure_dimension):
      self.kf = KalmanFilter(dim_x=state_dimension, dim_z=measure_dimension)

    def predict(self):
      self.kf.predict()

    def update(self, measurement):
      self.kf.update(measurement)

 

@app.route("/fuse", methods=["POST"])

def fuse():
    """Fuses sensor data, including accelerometer, gyroscope, latitude, and longitude,
    using a Kalman filter and returns a more accurate latitude and longitude.
    Returns:
        A JSON object containing the fused sensor data, including latitude and longitude.
    """

    # Get the sensor data from the request body
    sensor_data = request.get_json()

    # Create a Kalman filter object for sensor fusion
    kalman_filter = SensorFusionKalmanFilter(state_dimension=6, measure_dimension=6)

    # Extract accelerometer and gyroscope data
    accelerometer_reading = np.array(sensor_data["accelerometer"]["reading"])

    gyroscope_reading = np.array(sensor_data["gyroscope"]["reading"])

    # Concatenate accelerometer and gyroscope readings for sensor fusion
    measurement = np.concatenate((accelerometer_reading, gyroscope_reading))

    # Predict the next state
    kalman_filter.predict()

    # Update the Kalman filter with the sensor measurement
    kalman_filter.update(measurement)

    # Extract latitude and longitude data
    latitude_readings = [sensor["latitude"] for sensor in sensor_data["sensors"]]
    longitude_readings = [sensor["longitude"] for sensor in sensor_data["sensors"]]

    # Fuse latitude and longitude data
    fused_latitude = np.mean(latitude_readings)
    fused_longitude = np.mean(longitude_readings)

    # Get the state estimate from the Kalman filter
    state_estimate = kalman_filter.kf.x

    # Update the fused sensor data with the fused latitude and longitude
    state_estimate[0] = fused_latitude
    state_estimate[1] = fused_longitude

    # Return the fused sensor data, including latitude and longitude
    return jsonify({
      "fused_reading": state_estimate.tolist()
    })

if __name__ == "__main__":
    app.run()