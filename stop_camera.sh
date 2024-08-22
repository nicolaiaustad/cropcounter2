#!/bin/bash
echo "Stopping camera script..." >> /home/nicolaiaustad/Desktop/CropCounter/stop_camera.log
rm /tmp/usb_inserted
sleep 10  # Optional delay to ensure the Python script detects the file removal
pkill -SIGINT -f /home/nicolaiaustad/Desktop/CropCounter/run.py  # Adjust the signal and script name as necessary
if systemctl is-active --quiet camera_script.service; then
    systemctl stop camera_script.service
    echo "Camera script service stopped." >> /home/nicolaiaustad/Desktop/CropCounter/stop_camera.log
else
    echo "No running program found to stop." >> /home/nicolaiaustad/Desktop/CropCounter/stop_camera.log
fi
