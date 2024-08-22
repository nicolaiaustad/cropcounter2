#!/bin/bash 
set -e 
# Copy udev rules 
sudo cp config/99-usb-stick.rules /etc/udev/rules.d/ 
# Copy systemd service files 
sudo cp config/camera_script.service /etc/systemd/system/ 
# Reload udev rules 
sudo udevadm control --reload-rules 
sudo udevadm trigger 
# Enable and start systemd services 
sudo systemctl daemon-reload 
sudo systemctl enable camera_script.service 
sudo systemctl start camera_script.service 
echo "Configuration setup complete." 
