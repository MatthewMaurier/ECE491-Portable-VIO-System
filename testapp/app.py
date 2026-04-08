from flask import Flask, render_template, jsonify, request, send_file, abort
import subprocess
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud, BatteryState
#from mavros_msgs.msg import State
import threading
import base64
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import signal
import sys
import psutil
import socket
import math
import os
import time
import csv
from urllib.parse import quote
from datetime import datetime, timezone
from ina219 import INA219, DeviceRangeError

app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True
odom_data = {}
image_data = None
bridge = CvBridge()
total_path_length = 0.0
last_position = None
features_count = 0
health_status = "Healthy"
health_reason = ""
system_rebooting = False
send_image = False
VIO_SERVICE = "openvins_ov9281_mpu_system.service"
log_enabled = False
log_lock = threading.Lock()
log_file_path = None
log_file_handle = None
log_csv_writer = None
#VIO_SERVICE="vio_ov9281_mpu_system.service"
mav_battery_data = {}
ina219_data = {
    'voltage': None,
    'current': None,
    'power': None,
    'shunt_voltage': None,
    'battery_percent': None,
    'remaining_capacity_mah': None,
    'status': 'Not started'
}

ina219_lock = threading.Lock()

SHUNT_OHMS = 0.01
MAX_EXPECTED_AMPS = 3.0
BATTERY_CAPACITY_MAH = 10000
remaining_capacity = BATTERY_CAPACITY_MAH
INA219_BUS = 22          # change to 3 later if you move back to software I2C
INA219_ADDRESS = 0x40   # default INA219 address
vins_process = None

initial_roll = None
initial_pitch = None
initial_yaw = None

initial_quaternion = None

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    if p1 is not None and p2 is not None:
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
    else:
        return 0.0

# Callback for odometry data

def get_ip():
    hostname = socket.gethostname()
    return socket.gethostbyname_ex(hostname)[2]





def odom_callback(msg):
    global odom_data, total_path_length, last_position
    global health_status, health_reason, initial_quaternion

    orientation = msg.pose.pose.orientation
    position = msg.pose.pose.position

    # Calculate traveled path length
    #if last_position is not None:
    #   step = calculate_distance(last_position, position)
    #   if step > 0.005:
     #       total_path_length += step
    #last_position = position

    odom_data = {
        'x': position.x,
        'y': position.y,
        'z': position.z,
        'roll': 0,
        'pitch': 0,
        'yaw': 0
    }

    import tf.transformations as tf
    import math

    q_current = [orientation.x, orientation.y, orientation.z, orientation.w]

    # Save first quaternion as fixed reference
    if initial_quaternion is None:
        initial_quaternion = q_current[:]   # make a copy

    # Relative orientation
    q_init_inv = tf.quaternion_inverse(initial_quaternion)
    q_rel = tf.quaternion_multiply(q_init_inv, q_current)

    yaw_rel, pitch_rel, roll_rel  = tf.euler_from_quaternion(q_rel)

    odom_data['roll'] = roll_rel * 180.0 / math.pi
    odom_data['pitch'] = pitch_rel * 180.0 / math.pi
    odom_data['yaw'] = yaw_rel * 180.0 / math.pi 

    # Health check based on odometry values
    if abs(position.x) > 1000 or abs(position.y) > 1000 or abs(position.z) > 1000:
        health_status = "Unhealthy"
        health_reason = "Odometry values diverging"
    else:
        health_status = "Healthy"
        health_reason = ""
    
    global log_enabled, log_csv_writer, log_file_handle, log_file_path
    if log_enabled:
        with log_lock:
            if log_csv_writer is not None:
                # Use ROS time if /use_sim_time is enabled; otherwise wall time
                try:
                    t_ros = rospy.Time.now().to_sec()
                    unix_t = t_ros if t_ros and t_ros > 0 else time.time()
                except Exception:
                    unix_t = time.time()

                iso_t = datetime.fromtimestamp(unix_t, tz=timezone.utc).isoformat()
                log_csv_writer.writerow([
                    unix_t, iso_t,
                    position.x, position.y, position.z,
                    odom_data['roll'], odom_data['pitch'], odom_data['yaw']
                ])
                # Keep it safe if power dies / crash: flush each row (you can make this less frequent later)
                log_file_handle.flush()
def wrap_angle_rad(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi
# Callback for image data
def image_callback(msg):
    global image_data, send_image
    if not send_image:
        return
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        success, buffer = cv2.imencode('.jpg', cv_image)
        if not success:
            rospy.logwarn("Failed to encode image")
            return
        image_data = base64.b64encode(buffer).decode('utf-8')
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge error: {e}")
    except Exception as e:
        rospy.logerr(f"Error in image_callback: {e}")

# Callback for loop_features data
def loop_features_callback(msg):
    global features_count, health_status, health_reason
    features_count = len(msg.points)
    if features_count < 10:
        health_status = "Unhealthy"
        health_reason = "Low number of features"
    elif health_status != "Unhealthy":
        health_status = "Healthy"
        health_reason = ""

# Callback for MAVLink battery data
def mav_battery_callback(msg):
    global mav_battery_data
    mav_battery_data = {
        'voltage': msg.voltage,
        'current': msg.current,
        'percentage': msg.percentage*100.0,
        'status': msg.power_supply_status,
        'health': msg.power_supply_health
    }

def ros_listener():
    # Main VIO odom (use OpenVINS instead of MAVROS)
    rospy.Subscriber('/ov_msckf/odomimu', Odometry, odom_callback)

    # Camera + features
    rospy.Subscriber('/cam0/image_raw', Image, image_callback)
    rospy.Subscriber('/ov_msckf/loop_feats', PointCloud, loop_features_callback)

    # Keep MAVROS telemetry endpoints
    rospy.Subscriber('/mavros/battery', BatteryState, mav_battery_callback)
    rospy.spin()

def _open_new_log():
    global log_file_path, log_file_handle, log_csv_writer
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Example filename: vio_pose_20260222_101530.csv
    fname = datetime.now().strftime("vio_pose_%Y%m%d_%H%M%S.csv")
    log_file_path = os.path.join(logs_dir, fname)

    log_file_handle = open(log_file_path, "w", newline="")
    log_csv_writer = csv.writer(log_file_handle)
    log_csv_writer.writerow(["unix_time", "iso_time", "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg"])
    log_file_handle.flush()

def _close_log():
    global log_file_handle, log_csv_writer
    try:
        if log_file_handle:
            log_file_handle.flush()
            log_file_handle.close()
    finally:
        log_file_handle = None
        log_csv_writer = None
        
def ina219_reader():
    global ina219_data, remaining_capacity

    try:
        ina = INA219(
            SHUNT_OHMS,
            MAX_EXPECTED_AMPS,
            address=INA219_ADDRESS,
            busnum=INA219_BUS
        )
        ina.configure(ina.RANGE_16V)

        last_time = time.time()

        with ina219_lock:
            ina219_data['status'] = 'Connected'

        while True:
            try:
                voltage = ina.voltage()              # V
                current = ina.current()              # mA
                power = ina.power()                  # mW
                shunt_voltage = ina.shunt_voltage()  # V

                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time

                used_mah = (current * dt) / 3600.0
                remaining_capacity -= used_mah

                if remaining_capacity < 0:
                    remaining_capacity = 0

                battery_percent = ((remaining_capacity / BATTERY_CAPACITY_MAH) * -100.0) + 100

                with ina219_lock:
                    ina219_data = {
                        'voltage': voltage,
                        'current': current,
                        'power': power,
                        'shunt_voltage': shunt_voltage,
                        'battery_percent': battery_percent,
                        'remaining_capacity_mah': remaining_capacity,
                        'status': 'Connected'
                    }

            except DeviceRangeError as e:
                with ina219_lock:
                    ina219_data['status'] = f'Range error: {e}'
            except Exception as e:
                with ina219_lock:
                    ina219_data['status'] = f'Read error: {e}'

            time.sleep(1)

    except Exception as e:
        with ina219_lock:
            ina219_data['status'] = f'Init error: {e}'
            
@app.route("/")
def index():
    ip = subprocess.check_output(["hostname", "-I"]).decode().strip()
    return render_template("index.html", ip=ip)
    
@app.route("/pi")
def pi():
    ip = subprocess.check_output(["hostname", "-I"]).decode().strip()
    return render_template("pi.html", ip=ip)

@app.route('/odom')
def get_odom():
    return jsonify(odom_data)

@app.route('/image')
def get_image():
    return jsonify({'image': image_data})

@app.route('/update_image_status', methods=['POST'])
def update_image_status():
    global send_image
    data = request.json
    send_image = data.get('send_image', False)
    return jsonify({'status': 'success'})

@app.route('/restart_vio', methods=['POST'])
def restart_vio():
    global total_path_length, last_position, odom_data, features_count, health_status, health_reason
    global initial_roll, initial_pitch, initial_yaw
    initial_roll = None
    initial_pitch = None
    initial_yaw = None
    # Reset variables
    total_path_length = 0.0
    last_position = None
    odom_data = {}
    features_count = 0
    health_status = "Healthy"
    health_reason = ""
    # Restart VIO service
    subprocess.call(['sudo', 'systemctl', 'restart', VIO_SERVICE])
    return '', 204

@app.route('/reboot_system', methods=['POST'])
def reboot_system():
    global system_rebooting, total_path_length, last_position, odom_data, features_count, health_status, health_reason
    # Reset variables
    system_rebooting = True
    total_path_length = 0.0
    last_position = None
    odom_data = {}
    features_count = 0
    health_status = "Healthy"
    health_reason = ""
    # Reboot system
    subprocess.call(['sudo', 'reboot'])
    return '', 204

@app.route('/shutdown_system', methods=['POST'])
def shutdown_system():
    global system_rebooting, total_path_length, last_position, odom_data, features_count, health_status, health_reason
    # Reset variables
    system_rebooting = True
    total_path_length = 0.0
    last_position = None
    odom_data = {}
    features_count = 0
    health_status = "Healthy"
    health_reason = ""
    # Shutdown system
    subprocess.call(['sudo', 'shutdown', 'now'])
    return '', 204

@app.route('/system_stats')
def system_stats():
    global system_rebooting
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent

    # Fetch temperature using vcgencmd
    temp_output = subprocess.check_output(['vcgencmd', 'measure_temp']).decode()
    temp = float(temp_output.split('=')[1].split('\'')[0])

    return jsonify({'cpu_usage': cpu_usage, 'memory_usage': memory_usage, 'temperature': temp, 'rebooting': system_rebooting})

@app.route('/features_count')
def get_features_count():
    return jsonify({'features_count': features_count})

@app.route('/health_status')
def get_health_status():
    # Check if OpenVINS odom topic is publishing
    try:
        rospy.wait_for_message('/ov_msckf/odomimu', Odometry, timeout=1.0)
        return jsonify({'health_status': health_status, 'health_reason': health_reason})
    except:
        return jsonify({'health_status': 'Unhealthy', 'health_reason': 'OpenVINS not publishing'})

@app.route('/path_length')
def get_path_length():
    global total_path_length
    if total_path_length is None or math.isnan(total_path_length):
        total_path_length = 0.0
    return jsonify({'path_length': total_path_length})

@app.route('/mav_battery')
def get_mav_battery():
    with ina219_lock:
        return jsonify({
            'voltage': ina219_data.get('voltage'),
            'current': ina219_data.get('current'),
            'percentage': ina219_data.get('battery_percent'),
            'status': ina219_data.get('status'),
            'health': 'INA219'
        })
    
@app.route('/ina219_battery')
def get_ina219_battery():
    with ina219_lock:
        return jsonify(ina219_data)

@app.route('/log_status')
def log_status():
    return jsonify({
        'enabled': log_enabled,
        'file': os.path.basename(log_file_path) if log_file_path else None
    })

@app.route('/start_log', methods=['POST'])
def start_log():
    global log_enabled
    with log_lock:
        if not log_enabled:
            _open_new_log()
            log_enabled = True
    return jsonify({'status': 'started', 'file': os.path.basename(log_file_path)})

@app.route('/stop_log', methods=['POST'])
def stop_log():
    global log_enabled
    with log_lock:
        if log_enabled:
            log_enabled = False
            _close_log()
    return jsonify({'status': 'stopped', 'file': os.path.basename(log_file_path) if log_file_path else None})

def _logs_dir():
    return os.path.join(os.path.dirname(__file__), "logs")

@app.route("/start_vins", methods=["POST"])
def start_vins():
    global vins_process, total_path_length, initial_roll, initial_pitch, initial_yaw
    if vins_process is None or vins_process.poll() is not None:
        # Start Vins
        initial_roll = None
        initial_pitch = None
        initial_yaw = None
        total_path_length = 0
        vins_process = subprocess.Popen(
        ['/home/vin/start_vins.sh'],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        preexec_fn=os.setsid
        )
        return jsonify({"status": "started"})
    else:
        return jsonify({"status": "already running"})

@app.route("/stop_vins", methods=["POST"])
def stop_vins():
    global vins_process
    #if vins_process is not None and vins_process.poll() is None:
    # Kill the process group
    os.killpg(os.getpgid(vins_process.pid), signal.SIGTERM)
    subprocess.run(["pkill","-f","ov_msckf"])
    subprocess.run(["pkill", "-f", "ism330publisher.py"], check=False)
    vins_process = None
    return jsonify({"status": "stopped"})
    #else:
        #return jsonify({"status": "not running"})

@app.route("/vins_status")
def vins_status():
    result = subprocess.run(
        ["systemctl", "is-active", "vins"],
        capture_output=True,
        text=True
    )
    status = result.stdout.strip()
    return jsonify({"status": status})

@app.route('/download_log')
def browse_logs():
    logs_dir = _logs_dir()
    os.makedirs(logs_dir, exist_ok=True)

    files = []
    for fn in os.listdir(logs_dir):
        if fn.endswith(".csv") and fn.startswith("vio_pose_"):
            full = os.path.join(logs_dir, fn)
            st = os.stat(full)
            files.append({
                "name": fn,
                "mtime": st.st_mtime,
                "mtime_str": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "size_kb": st.st_size / 1024.0
            })

    files.sort(key=lambda f: f["mtime"], reverse=True)

    html = """
    <html>
    <head><title>VIO Logs</title></head>
    <body>
    <h2>VIO Logs</h2>
    <p><a href="/">⬅ Back to Home</a></p>
    <p><a href="/pi">⬅ Back to Pi GUI</a></p>
    <table border="1" cellpadding="5">
    <tr><th>Filename</th><th>Modified</th><th>Size (KB)</th></tr>
    """

    for f in files:
        link = f"/download_log/file/{quote(f['name'])}"
        html += f"<tr><td><a href='{link}'>{f['name']}</a></td><td>{f['mtime_str']}</td><td>{f['size_kb']:.1f}</td></tr>"

    html += "</table></body></html>"
    return html


@app.route('/download_log/file/<path:filename>')
def download_log_file(filename):
    logs_dir = _logs_dir()
    full = os.path.abspath(os.path.join(logs_dir, filename))

    if not full.startswith(os.path.abspath(logs_dir)):
        abort(400)

    if not os.path.exists(full):
        abort(404)

    return send_file(full, as_attachment=True)


def run_flask_app():
    # Show request logs + full tracebacks in terminal
    app.logger.setLevel("DEBUG")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

def signal_handler(sig, frame):
    print('Shutting down...')
    rospy.signal_shutdown('Signal received')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    rospy.init_node('web_interface_node', anonymous=True)
    ros_thread = threading.Thread(target=ros_listener)
    ros_thread.start()

    ina_thread = threading.Thread(target=ina219_reader, daemon=True)
    ina_thread.start()
    
    
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.start()

    ros_thread.join()
    flask_thread.join()
