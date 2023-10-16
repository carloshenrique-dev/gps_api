import numpy as np
import sys
import json
from flask import Flask, request, jsonify

# Create a Flask app
app = Flask(__name__)

np.random.seed(777)

sys.path.append('./src')

from kalman_filters import ExtendedKalmanFilter as EKF
from utils import lla_to_enu, enu_to_lla ,normalize_angles, quaternion, madgwickahrs, savitzky_golay_filter

mad_filter = madgwickahrs.MadgwickAHRS(sampleperiod=1, quaternion=quaternion.Quaternion(1, 0, 0, 0), beta=1, zeta=0)


# Renaming and joining
gyro = []
accel = []
magne = []
yaws = []
yaw_rates = []
velocities = []
trajectory = []
yaw_rate_noise_std = 0
initial_yaw_std = 0


# Inside the `/sensor_data` route
@app.route('/sensor_data', methods=['POST'])
def receive_sensor_data():
    data = request.get_json()

    gyro.append(data.get('gyroscope', [0, 0, 0]))
    accel.append(data.get('accelerometer', [0, 0, 0]))
    velocities.append([data.get('speed', 0) if data.get(
        'speed', 0) > 0 else 0, data.get('speedAccuracy', 0) if data.get(
        'speedAccuracy', 0) > 0 else 0])
    trajectory.append([data.get('latitude', 0), data.get('longitude', 0), data.get('altitude', 0), data.get('timestamp', 0), data.get('gpsAccuracy', 0)])

    mad_filter.update_imu(gyro[-1], accel[-1])

    if len(trajectory) < 2:
        # Not enough data points for estimation
        response = {
            'estimated_x': 0,
            'estimated_y': 0,
            'estimated_yaw': 0,
        }
        return jsonify(response)

    roll, pitch, yaw = mad_filter.quaternion.to_euler_angles()

    if len(yaws) > 0:
        last_yaw = yaws[-1]
    else:
        last_yaw = yaw

    yaw_rate = yaw - last_yaw
    yaws.append(yaw)
    yaw_rates.append(yaw_rate)

    trajectoryArray = np.array(trajectory).T
    yawsArray = np.array(yaws)
    yaw_rate = np.array(yaw_rates)
    velocitiesArray = np.array(velocities)

    # Transform GPS trajectory from [lon, lat, alt] to local [x, y, z] coord so that Kalman filter can handle it.
    origin = trajectoryArray[:, 0]  # set the initial position to the origin
    gt_trajectory_xyz = lla_to_enu(trajectoryArray, origin)

    xs, ys, _ = gt_trajectory_xyz

    # Add noise to GPS data
    #xy_obs_noise_std = 1.0  # Adjust noise level as needed
    #xy_obs_noise = np.random.normal(0.0, xy_obs_noise_std, (2, 1))  # Assuming one measurement

    #obs_trajectory_xyz = gt_trajectory_xyz.copy()
    #obs_trajectory_xyz[:2, -1] += xy_obs_noise.flatten()

    # Prepare initial estimate and error covariance
    #initial_yaw_std = np.pi
    initial_yaw = yawsArray[0] #np.random.normal(0, initial_yaw_std)

    x = np.array([
        gt_trajectory_xyz[0, -1],
        gt_trajectory_xyz[1, -1],
        initial_yaw
    ])

    xy_obs_noise_std = trajectory[-1][4]

    # Covariance for initial state estimation error (Sigma_0)
    P = np.array([
        [1 ** 2., 0., 0.],
        [0., 1 ** 2., 0.],
        [0., 0., 1 ** 2.]
    ])

    # Prepare measurement error covariance Q
    Q = np.array([
        [1 ** 2., 0.],
        [0., 1 ** 2.]
    ])

    forward_velocity_noise_std = velocities[-1][1]

    # Prepare state transition noise covariance R
    R = np.array([
        [1 ** 2., 0., 0.],
        [0., 1 ** 2., 0.],
        [0., 0., 1 ** 2.]
    ])

    # Initialize Kalman filter
    kf = EKF(x, P)

    # Arrays to store estimated pose and variance
    mu_x = [x[0]]
    mu_y = [x[1]]
    mu_theta = [x[2]]

    var_x = [P[0, 0]]
    var_y = [P[1, 1]]
    var_theta = [P[2, 2]]

    t = trajectory[-1][3]
    dt = t - trajectory[-2][3]

    # Get control input `u = [v, omega]`
    u = np.array([
        velocitiesArray[-1][0],
        yaw_rate[-1]
    ])

    # Propagate
    kf.propagate(u, dt, R)

    # Get measurement `z = [x, y]`
    z = np.array([
        gt_trajectory_xyz[0, -1],
        gt_trajectory_xyz[1, -1]
    ])

    # Update
    kf.update(z, Q)

    # Save estimated state
    mu_x.append(kf.x[0])
    mu_y.append(kf.x[1])
    mu_theta.append(normalize_angles(kf.x[2]))

    # Save estimated variance
    var_x.append(kf.P[0, 0])
    var_y.append(kf.P[1, 1])
    var_theta.append(kf.P[2, 2])

    mu_x = np.array(mu_x)
    mu_y = np.array(mu_y)
    mu_theta = np.array(mu_theta)

    var_x = np.array(var_x)
    var_y = np.array(var_y)
    var_theta = np.array(var_theta)

    # Get the estimated position and orientation
    estimated_x, estimated_y, estimated_yaw = kf.x

    result = enu_to_lla([estimated_x, estimated_y, 0],origin)

    estimated_location = [float(x) for x in result.T[-1].tolist()]

    # Return the results as a JSON response
    response = {
        'estimated_x': estimated_x,
        'estimated_y': estimated_y,
        'estimated_yaw': estimated_yaw,
        'estimated_location': estimated_location,
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run('192.168.15.62')
