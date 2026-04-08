#!/bin/bash
# start vins
lxterminal -e "bash -i -c 'python3 ism330publisher.py'" & 
sleep 2
lxterminal -e "bash -i -c 'roslaunch -v ov_msckf subscribe2.launch :=config=test7'" &

