#!/bin/bash 
sudo apt update 
sudo apt install -y python3-picamera2 python3-pillow python3-numpy python3-pandas python3-pyproj python3-geopandas python3-matplotlib python3-seaborn python3-inotify python3-opencv python3-shapely python3-fiona python3-logging
# Install Python dependencies
pip install -r requirements.txt
