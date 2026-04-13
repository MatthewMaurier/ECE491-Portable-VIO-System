## Project Files and Setup

This folder contains all files modified from a clean installation of **ROS1** and **OpenVINS** on a **Raspberry Pi 4** for this project.

### File Locations

Place the files in the following locations:

- **`hardware_init.sh`**  
  Add this to `~/.bashrc` so it runs on startup.

- **`start_vins.sh`**  
  Place this in `~/`

- **`testapp/`**  
  Place this in `~/src/roboeye/`

- **`open_vins/`** and **`camera_ros/`**  
  Place these in `~/catkin_ws/`

- **`ism330publisher.py`**  
  Place this in `~/src/`

- **`capstone desktop app/`**  
  Place this on the desktop of the local machine.

### Additional Project Files

Also included are:

- **Altium source files** in a `.zip` archive
- **Enclosure design file** in `.f3z` format

### Calibration

To calibrate the device, refer to the folder named **`Calibration`**.

### Usage

To use the device in its current state, read **`Quick Start Guide.pdf`**.

### Notes

With these files placed in the correct locations, the project should run on any **Raspberry Pi 4** with **ROS1** and **OpenVINS** installed.

**Contribution:** Everybody