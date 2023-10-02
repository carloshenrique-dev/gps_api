import numpy as np
import numbers
import warnings
from numpy.linalg import norm

class KalmanFilter:
    def __init__(self, state_dimension, measure_dimension, control_dimension):
        self.F = np.zeros((state_dimension, state_dimension)) # state transition model
        self.H = np.zeros((measure_dimension, state_dimension)) # observation model
        self.B = np.zeros((state_dimension, control_dimension)) # control matrix
        self.Q = np.zeros((state_dimension, state_dimension)) # process noise covariance
        self.R = np.zeros((measure_dimension, measure_dimension)) # observation noise covariance

        self.Uk = np.zeros((control_dimension, 1)) # control vector
        self.Zk = np.zeros((measure_dimension, 1)) # actual values (measured)

        self.Xk_km1 = np.zeros((state_dimension, 1)) # predicted state estimate
        self.Pk_km1 = np.zeros((state_dimension, state_dimension)) # predicted estimate covariance

        self.Yk = np.zeros((measure_dimension, 1)) # measurement innovation
        self.Sk = np.zeros((measure_dimension, measure_dimension)) # innovation covariance
        self.SkInv = np.zeros((measure_dimension, measure_dimension)) # innovation covariance inverse

        self.K = np.zeros((state_dimension, measure_dimension)) # Kalman gain (optimal)
        self.Xk_k = np.zeros((state_dimension, 1)) # updated (current) state
        self.Pk_k = np.zeros((state_dimension, state_dimension)) # updated estimate covariance
        self.Yk_k = np.zeros((measure_dimension, 1)) # post fit residual

        self.auxBxU = np.zeros((state_dimension, 1))
        self.auxSDxSD = np.zeros((state_dimension, state_dimension))
        self.auxSDxMD = np.zeros((state_dimension, measure_dimension))

    def predict(self):
        # Xk|k-1 = Fk*Xk-1|k-1 + Bk*Uk
        self.Xk_km1 = np.dot(self.F, self.Xk_k) + np.dot(self.B, self.Uk)

        # Pk|k-1 = Fk*Pk-1|k-1*Fk(t) + Qk
        self.Pk_km1 = np.dot(self.F, np.dot(self.Pk_k, self.F.T)) + self.Q

    def update(self):
        # Yk = Zk - Hk*Xk|k-1
        self.Yk = self.Zk - np.dot(self.H, self.Xk_km1)

        # Sk = Rk + Hk*Pk|k-1*Hk(t)
        self.Sk = self.R + np.dot(self.H, np.dot(self.Pk_km1, self.H.T))

        # Kk = Pk|k-1*Hk(t)*Sk(inv)
        self.SkInv = np.linalg.inv(self.Sk)
        self.K = np.dot(self.Pk_km1, np.dot(self.H.T, self.SkInv))

        # xk|k = xk|k-1 + Kk*Yk
        self.Xk_k = self.Xk_km1 + np.dot(self.K, self.Yk)

        # Pk|k = (I - Kk*Hk) * Pk|k-1
        self.Pk_k = (np.identity(self.Pk_km1.shape[0]) - np.dot(self.K, self.H)) * self.Pk_km1


class MadgwickAHRS:
    samplePeriod = 1/256
    quaternion = Quaternion(1, 0, 0, 0)
    beta = 1
    zeta = 0

    def __init__(self, sampleperiod=None, quaternion=None, beta=None, zeta=None):
        """
        Initialize the class with the given parameters.
        :param sampleperiod: The sample period
        :param quaternion: Initial quaternion
        :param beta: Algorithm gain beta
        :param beta: Algorithm gain zeta
        :return:
        """
        if sampleperiod is not None:
            self.samplePeriod = sampleperiod
        if quaternion is not None:
            self.quaternion = quaternion
        if beta is not None:
            self.beta = beta
        if zeta is not None:
            self.zeta = zeta

    def update(self, gyroscope, accelerometer, magnetometer):
        """
        Perform one update step with data from a AHRS sensor array
        :param gyroscope: A three-element array containing the gyroscope data in radians per second.
        :param accelerometer: A three-element array containing the accelerometer data. Can be any unit since a normalized value is used.
        :param magnetometer: A three-element array containing the magnetometer data. Can be any unit since a normalized value is used.
        :return:
        """
        q = self.quaternion

        gyroscope = np.array(gyroscope, dtype=float).flatten()
        accelerometer = np.array(accelerometer, dtype=float).flatten()
        magnetometer = np.array(magnetometer, dtype=float).flatten()

        # Normalise accelerometer measurement
        if norm(accelerometer) is 0:
            warnings.warn("accelerometer is zero")
            return
        accelerometer /= norm(accelerometer)

        # Normalise magnetometer measurement
        if norm(magnetometer) is 0:
            warnings.warn("magnetometer is zero")
            return
        magnetometer /= norm(magnetometer)

        h = q * (Quaternion(0, magnetometer[0], magnetometer[1], magnetometer[2]) * q.conj())
        b = np.array([0, norm(h[1:3]), 0, h[3]])

        # Gradient descent algorithm corrective step
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - accelerometer[0],
            2*(q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
            2*(0.5 - q[1]**2 - q[2]**2) - accelerometer[2],
            2*b[1]*(0.5 - q[2]**2 - q[3]**2) + 2*b[3]*(q[1]*q[3] - q[0]*q[2]) - magnetometer[0],
            2*b[1]*(q[1]*q[2] - q[0]*q[3]) + 2*b[3]*(q[0]*q[1] + q[2]*q[3]) - magnetometer[1],
            2*b[1]*(q[0]*q[2] + q[1]*q[3]) + 2*b[3]*(0.5 - q[1]**2 - q[2]**2) - magnetometer[2]
        ])
        j = np.array([
            [-2*q[2],                  2*q[3],                  -2*q[0],                  2*q[1]],
            [2*q[1],                   2*q[0],                  2*q[3],                   2*q[2]],
            [0,                        -4*q[1],                 -4*q[2],                  0],
            [-2*b[3]*q[2],             2*b[3]*q[3],             -4*b[1]*q[2]-2*b[3]*q[0], -4*b[1]*q[3]+2*b[3]*q[1]],
            [-2*b[1]*q[3]+2*b[3]*q[1], 2*b[1]*q[2]+2*b[3]*q[0], 2*b[1]*q[1]+2*b[3]*q[3],  -2*b[1]*q[0]+2*b[3]*q[2]],
            [2*b[1]*q[2],              2*b[1]*q[3]-4*b[3]*q[1], 2*b[1]*q[0]-4*b[3]*q[2],  2*b[1]*q[1]]
        ])
        step = j.T.dot(f)
        step /= norm(step)  # normalise step magnitude

        # Gyroscope compensation drift
        gyroscopeQuat = Quaternion(0, gyroscope[0], gyroscope[1], gyroscope[2])
        stepQuat = Quaternion(step.T[0], step.T[1], step.T[2], step.T[3])

        gyroscopeQuat = gyroscopeQuat + (q.conj() * stepQuat) * 2 * self.samplePeriod * self.zeta * -1

        # Compute rate of change of quaternion
        qdot = (q * gyroscopeQuat) * 0.5 - self.beta * step.T

        # Integrate to yield quaternion
        q += qdot * self.samplePeriod
        self.quaternion = Quaternion(q / norm(q))  # normalise quaternion

    def update_imu(self, gyroscope, accelerometer):
        """
        Perform one update step with data from a IMU sensor array
        :param gyroscope: A three-element array containing the gyroscope data in radians per second.
        :param accelerometer: A three-element array containing the accelerometer data. Can be any unit since a normalized value is used.
        """
        q = self.quaternion

        gyroscope = np.array(gyroscope, dtype=float).flatten()
        accelerometer = np.array(accelerometer, dtype=float).flatten()

        # Normalise accelerometer measurement
        if norm(accelerometer) is 0:
            warnings.warn("accelerometer is zero")
            return
        accelerometer /= norm(accelerometer)

        # Gradient descent algorithm corrective step
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - accelerometer[0],
            2*(q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
            2*(0.5 - q[1]**2 - q[2]**2) - accelerometer[2]
        ])
        j = np.array([
            [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
            [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
            [0, -4*q[1], -4*q[2], 0]
        ])
        step = j.T.dot(f)
        step /= norm(step)  # normalise step magnitude

        # Compute rate of change of quaternion
        qdot = (q * Quaternion(0, gyroscope[0], gyroscope[1], gyroscope[2])) * 0.5 - self.beta * step.T

        # Integrate to yield quaternion
        q += qdot * self.samplePeriod
        self.quaternion = Quaternion(q / norm(q))  # normalise quaternion

class Quaternion:
    """
    A simple class implementing basic quaternion arithmetic.
    """
    def __init__(self, w_or_q, x=None, y=None, z=None):
        """
        Initializes a Quaternion object
        :param w_or_q: A scalar representing the real part of the quaternion, another Quaternion object or a
                    four-element array containing the quaternion values
        :param x: The first imaginary part if w_or_q is a scalar
        :param y: The second imaginary part if w_or_q is a scalar
        :param z: The third imaginary part if w_or_q is a scalar
        """
        self._q = np.array([1, 0, 0, 0])

        if x is not None and y is not None and z is not None:
            w = w_or_q
            q = np.array([w, x, y, z])
        elif isinstance(w_or_q, Quaternion):
            q = np.array(w_or_q.q)
        else:
            q = np.array(w_or_q)
            if len(q) != 4:
                raise ValueError("Expecting a 4-element array or w x y z as parameters")

        self.q = q

    # Quaternion specific interfaces

    def conj(self):
        """
        Returns the conjugate of the quaternion
        :rtype : Quaternion
        :return: the conjugate of the quaternion
        """
        return Quaternion(self._q[0], -self._q[1], -self._q[2], -self._q[3])

    def to_angle_axis(self):
        """
        Returns the quaternion's rotation represented by an Euler angle and axis.
        If the quaternion is the identity quaternion (1, 0, 0, 0), a rotation along the x axis with angle 0 is returned.
        :return: rad, x, y, z
        """
        if self[0] == 1 and self[1] == 0 and self[2] == 0 and self[3] == 0:
            return 0, 1, 0, 0
        rad = np.arccos(self[0]) * 2
        imaginary_factor = np.sin(rad / 2)
        if abs(imaginary_factor) < 1e-8:
            return 0, 1, 0, 0
        x = self._q[1] / imaginary_factor
        y = self._q[2] / imaginary_factor
        z = self._q[3] / imaginary_factor
        return rad, x, y, z

    @staticmethod
    def from_angle_axis(rad, x, y, z):
        s = np.sin(rad / 2)
        return Quaternion(np.cos(rad / 2), x*s, y*s, z*s)

    def to_euler_angles(self):
        pitch = np.arcsin(2 * self[1] * self[2] + 2 * self[0] * self[3])
        if np.abs(self[1] * self[2] + self[3] * self[0] - 0.5) < 1e-8:
            roll = 0
            yaw = 2 * np.arctan2(self[1], self[0])
        elif np.abs(self[1] * self[2] + self[3] * self[0] + 0.5) < 1e-8:
            roll = -2 * np.arctan2(self[1], self[0])
            yaw = 0
        else:
            roll = np.arctan2(2 * self[0] * self[1] - 2 * self[2] * self[3], 1 - 2 * self[1] ** 2 - 2 * self[3] ** 2)
            yaw = np.arctan2(2 * self[0] * self[2] - 2 * self[1] * self[3], 1 - 2 * self[2] ** 2 - 2 * self[3] ** 2)
        return roll, pitch, yaw

    def to_euler123(self):
        roll = np.arctan2(-2 * (self[2] * self[3] - self[0] * self[1]), self[0] ** 2 - self[1] ** 2 - self[2] ** 2 + self[3] ** 2)
        pitch = np.arcsin(2 * (self[1] * self[3] + self[0] * self[1]))
        yaw = np.arctan2(-2 * (self[1] * self[2] - self[0] * self[3]), self[0] ** 2 + self[1] ** 2 - self[2] ** 2 - self[3] ** 2)
        return roll, pitch, yaw

    def __mul__(self, other):
        """
        multiply the given quaternion with another quaternion or a scalar
        :param other: a Quaternion object or a number
        :return:
        """
        if isinstance(other, Quaternion):
            w = self._q[0]*other._q[0] - self._q[1]*other._q[1] - self._q[2]*other._q[2] - self._q[3]*other._q[3]
            x = self._q[0]*other._q[1] + self._q[1]*other._q[0] + self._q[2]*other._q[3] - self._q[3]*other._q[2]
            y = self._q[0]*other._q[2] - self._q[1]*other._q[3] + self._q[2]*other._q[0] + self._q[3]*other._q[1]
            z = self._q[0]*other._q[3] + self._q[1]*other._q[2] - self._q[2]*other._q[1] + self._q[3]*other._q[0]

            return Quaternion(w, x, y, z)
        elif isinstance(other, numbers.Number):
            q = self._q * other
            return Quaternion(q)

    def __add__(self, other):
        """
        add two quaternions element-wise or add a scalar to each element of the quaternion
        :param other:
        :return:
        """
        if not isinstance(other, Quaternion):
            if len(other) != 4:
                raise TypeError("Quaternions must be added to other quaternions or a 4-element array")
            q = self._q + other
        else:
            q = self._q + other._q

        return Quaternion(q)

    # Implementing other interfaces to ease working with the class

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q):
        self._q = q

    def __getitem__(self, item):
        return self._q[item]

    def __array__(self):
        return self._q

class GeoHash:
    @staticmethod
    def interleave(x, y):
        x = (x | (x << 16)) & 0x0000ffff0000ffff
        x = (x | (x << 8)) & 0x00ff00ff00ff00ff
        x = (x | (x << 4)) & 0x0f0f0f0f0f0f0f0f
        x = (x | (x << 2)) & 0x3333333333333333
        x = (x | (x << 1)) & 0x5555555555555555

        y = (y | (y << 16)) & 0x0000ffff0000ffff
        y = (y | (y << 8)) & 0x00ff00ff00ff00ff
        y = (y | (y << 4)) & 0x0f0f0f0f0f0f0f0f
        y = (y | (y << 2)) & 0x3333333333333333
        y = (y | (y << 1)) & 0x5555555555555555

        return x | (y << 1)

    @staticmethod
    def encode_u64(lat, lon, prec):
        lat = lat / 180.0 + 1.5
        lon = lon / 360.0 + 1.5
        ilat = int(lat * (2 ** 52))
        ilon = int(lon * (2 ** 52))
        ilat >>= 20
        ilon >>= 20
        ilat &= 0x00000000ffffffff
        ilon &= 0x00000000ffffffff
        geohash = GeoHash.interleave(ilat, ilon) >> (GeoHash.GEOHASH_MAX_PRECISION - prec) * 5
        return geohash

    base32Table = [
        '0', '1', '2', '3', '4', '5', '6', '7',
        '8', '9', 'b', 'c', 'd', 'e', 'f', 'g',
        'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r',
        's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
    ]

    GEOHASH_MAX_PRECISION = 12

    @staticmethod
    def geohash_str(geohash, prec):
        buff = []
        geohash >>= 4  # porque não precisamos dos últimos 4 bits
        geohash &= 0x0fffffffffffffff  # não precisamos do sinal aqui
        while prec > 0:
            buff.append(GeoHash.base32Table[int(geohash & 0x1f)])
            geohash >>= 5
            prec -= 1
        return ''.join(reversed(buff))
    

class GPSAccKalmanFilter:
    def __init__(self, use_gps_speed, x, y, x_vel, y_vel, acc_dev, pos_dev, time_stamp_ms, vel_factor, pos_factor):
        mes_dim = 4 if use_gps_speed else 2
        self.use_gps_speed = use_gps_speed
        self.kf = KalmanFilter(4, mes_dim, 2)
        self.time_stamp_ms_predict = time_stamp_ms
        self.time_stamp_ms_update = time_stamp_ms
        self.acc_sigma = acc_dev
        self.predict_count = 0
        self.kf.Xk_k = np.array([[x], [y], [x_vel], [y_vel]])
        self.kf.H = np.eye(4)  # state has 4d and measurement has 4d too, so here is identity
        self.kf.Pk_k = np.eye(4) * pos_dev
        self.vel_factor = vel_factor
        self.pos_factor = pos_factor

    def rebuild_F(self, dt_predict):
        self.kf.F = np.array([
            [1.0, 0.0, dt_predict, 0.0],
            [0.0, 1.0, 0.0, dt_predict],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

    def rebuild_U(self, x_acc, y_acc):
        self.kf.Uk = np.array([[x_acc], [y_acc]])

    def rebuild_B(self, dt_predict):
        dt2 = 0.5 * dt_predict * dt_predict
        self.kf.B = np.array([
            [dt2, 0.0],
            [0.0, dt2],
            [dt_predict, 0.0],
            [0.0, dt_predict]
        ])

    def rebuild_R(self, pos_sigma, vel_sigma):
        pos_sigma *= self.pos_factor
        vel_sigma *= self.vel_factor

        if self.use_gps_speed:
            self.kf.R = np.array([
                [pos_sigma, 0.0, 0.0, 0.0],
                [0.0, pos_sigma, 0.0, 0.0],
                [0.0, 0.0, vel_sigma, 0.0],
                [0.0, 0.0, 0.0, vel_sigma]
            ])
        else:
            self.kf.R = np.eye(4) * pos_sigma

    def rebuild_Q(self, dt_update, acc_dev):
        vel_dev = acc_dev * self.predict_count
        pos_dev = vel_dev * self.predict_count / 2
        cov_dev = vel_dev * pos_dev

        pos_sig = pos_dev * pos_dev
        vel_sig = vel_dev * vel_dev

        self.kf.Q = np.array([
            [pos_sig, 0.0, cov_dev, 0.0],
            [0.0, pos_sig, 0.0, cov_dev],
            [cov_dev, 0.0, vel_sig, 0.0],
            [0.0, cov_dev, 0.0, vel_sig]
        ])

    def predict(self, time_now_ms, x_acc, y_acc):
        dt_predict = (time_now_ms - self.time_stamp_ms_predict) / 1000.0
        dt_update = (time_now_ms - self.time_stamp_ms_update) / 1000.0
        self.rebuild_F(dt_predict)
        self.rebuild_B(dt_predict)
        self.rebuild_U(x_acc, y_acc)

        self.predict_count += 1
        self.rebuild_Q(dt_update, self.acc_sigma)

        self.time_stamp_ms_predict = time_now_ms
        self.kf.predict()
        self.kf.Xk_km1 = self.kf.Xk_k.copy()

    def update(self, time_stamp, x, y, x_vel, y_vel, pos_dev, vel_err):
        self.predict_count = 0
        self.time_stamp_ms_update = time_stamp
        self.rebuild_R(pos_dev, vel_err)
        if self.use_gps_speed:
            self.kf.Zk = np.array([[x], [y], [x_vel], [y_vel]])
        else:
            self.kf.Zk = np.array([[x], [y]])
        self.kf.update()

    def get_current_x(self):
        return self.kf.Xk_k[0, 0]

    def get_current_y(self):
        return self.kf.Xk_k[1, 0]

    def get_current_x_vel(self):
        return self.kf.Xk_k[2, 0]

    def get_current_y_vel(self):
        return self.kf.Xk_k[3, 0]




