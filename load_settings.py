import os
import subprocess
import sys
import shutil
import logging

def check_permissions():
    if os.geteuid() != 0:
        print("This script must be run as root. Please use 'sudo'.")
        sys.exit(1)

def mount_usb(mount_point, device):
    # Check if the device is already mounted
    mount_output = subprocess.check_output(['mount']).decode()
    for line in mount_output.splitlines():
        if device in line:
            current_mount_point = line.split()[2]
            logging.info(f"{device} is already mounted at {current_mount_point}")
            # print(f"{device} is already mounted at {current_mount_point}")
            return current_mount_point
    
    os.makedirs(mount_point, exist_ok=True)
    subprocess.run(['sudo', 'mount', device, mount_point], check=True)
    return mount_point

def unmount_usb(mount_point):
    if os.path.ismount(mount_point):
        subprocess.run(['sudo', 'umount', mount_point], check=True)
    if os.path.exists(mount_point):
        shutil.rmtree(mount_point)

def copy_files(src, dst):
    if os.path.exists(src):
        if not os.path.exists(dst):
            os.makedirs(dst)
        subprocess.run(['sudo', 'cp', '-r', src, dst], check=True)

def read_settings_file(settings_file):
    settings = {}
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as file:
            for line in file:
                key, value = line.strip().split('=')
                settings[key] = value
    return settings

def copy_settings_and_shapefiles(mount_point, settings_dest, shapefiles_dest):
    settings_file = os.path.join(mount_point, 'settings.txt')
    shapefiles_folder = os.path.join(mount_point, 'shapefiles')
    
    # Copy settings file
    if os.path.exists(settings_file):
        shutil.copy(settings_file, settings_dest)
        # print(f"Copied {settings_file} to {settings_dest}")
        logging.info(f"Copied {settings_file} to {settings_dest}")
    else:
        logging.info(f"{settings_file} not found")
        # print(f"{settings_file} not found")
    
    # Copy shapefiles folder
    if os.path.exists(shapefiles_folder):
        shutil.copytree(shapefiles_folder, shapefiles_dest, dirs_exist_ok=True)
        logging.info(f"Copied {shapefiles_folder} to {shapefiles_dest}")
        # print(f"Copied {shapefiles_folder} to {shapefiles_dest}")
    else:
        logging.info(f"{shapefiles_folder} not found")
        # print(f"{shapefiles_folder} not found")

def find_shapefile(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".shp") and not file.startswith("._"):
                return os.path.join(root, file)
    return None

