#!/bin/bash
sleep 5  # Give some time for the USB to be fully ready
echo "Starting camera relay script..." >> /home/nicolaiaustad/Desktop/CropCounter/start_camera_relay.log
/home/nicolaiaustad/Desktop/CropCounter/start_camera.sh  
