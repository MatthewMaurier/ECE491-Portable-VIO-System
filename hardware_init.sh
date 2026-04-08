# Start ROS launch (this starts the master)
lxterminal -e "bash -i -c 'roslaunch libcamera_ros basic.launch'" &
sleep 5
# Now start IMU

# Now start camera publisher
lxterminal -e "bash -i -c 'python3 camerapublisher.py'" &
# Initialize website
sleep 5
lxterminal -e "bash -i -c 'python3 ~/src/roboeye/testapp/app.py'" &
# Run website
/usr/bin/chromium-browser --kiosk http://localhost:5000/pi

