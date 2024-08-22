#!/bin/bash
sleep 5  # Add a delay to ensure the USB drive is fully mounted
echo "Starting camera script..." >> /home/nicolaiaustad/Desktop/CropCounter/start_camera.log
touch /tmp/usb_inserted

if systemctl is-active --quiet camera_script.service; then
    echo "Program already running." >> /home/nicolaiaustad/Desktop/CropCounter/start_camera.log
else
    echo "Starting systemd service camera_script.service" >> /home/nicolaiaustad/Desktop/CropCounter/start_camera.log
    systemctl start camera_script.service
    echo "Camera script service started." >> /home/nicolaiaustad/Desktop/CropCounter/start_camera.log
fi
