import threading
import time
import serial
import adafruit_gps

# Set up the serial connection using pyserial
uart = serial.Serial("/dev/ttyUSB0", baudrate=9600, timeout=10)

# Create an instance of the GPS class
gps = adafruit_gps.GPS(uart, debug=False)

# Initialize the GPS module by sending commands to configure it
# Enable basic GGA and RMC info (most useful data)
gps.send_command(b"PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")
# Set update rate to once a second (1Hz)
gps.send_command(b"PMTK220,1000")

# Global variable to store the latest GPS data
latest_gps_data = {
    "timestamp": None,
    "latitude": None,
    "longitude": None,
    "fix_quality": None,
    "satellites": None,
    "speed_knots": None,
    "has_fix": False
}

def update_gps():
    global latest_gps_data
    while True:
        gps.update()
        if gps.has_fix:
            latest_gps_data = {
                "timestamp": "{}/{}/{} {:02}:{:02}:{:02}".format(
                    gps.timestamp_utc.tm_mon,
                    gps.timestamp_utc.tm_mday,
                    gps.timestamp_utc.tm_year,
                    gps.timestamp_utc.tm_hour,
                    gps.timestamp_utc.tm_min,
                    gps.timestamp_utc.tm_sec,
                ),
                "latitude": gps.latitude,
                "longitude": gps.longitude,
                "fix_quality": gps.fix_quality,
                "satellites": gps.satellites,
                "speed_knots": gps.speed_knots,
                "has_fix": gps.has_fix
            }
        time.sleep(1)

# Start the GPS update thread
gps_thread = threading.Thread(target=update_gps, daemon=True)
gps_thread.start()

def get_gps():
    global latest_gps_data
    
    if latest_gps_data["has_fix"] and (latest_gps_data["fix_quality"]==1 or latest_gps_data["fix_quality"]==2) :
        longitude = latest_gps_data["longitude"]
        latitude = latest_gps_data["latitude"]
        satellites= latest_gps_data["satellites"]
        speed = latest_gps_data["speed_knots"]
        timestamp = latest_gps_data["timestamp"]
        return longitude, latitude, satellites, speed
    else:
        return 0,0, latest_gps_data["satellites"], 999  #Long and lat = 0  means not valid gps_coordiantes retrieved

