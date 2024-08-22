import io
import logging
import socketserver
from http import server
from threading import Condition
import signal
import sys
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

PAGE = """\
<html>
<head>
<style>html, body { margin: 0; padding: 0; }</style>
<title>picamera2 MJPEG streaming demo</title>
</head>
<body>
<img src="stream.mjpg" width="1440" height="1080" style="height: 100%;" />
</body>
</html>
"""

logging.basicConfig(level=logging.DEBUG)

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning('Removed streaming client %s: %s', self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

def signal_handler(sig, frame):
    logging.info("Signal received, stopping camera...")
    picam2.stop_recording()
    logging.info("Camera recording stopped.")
    sys.exit(0)

logging.info("Starting camera setup...")
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (3280, 2464)}))
picam2.set_controls({
    "AnalogueGain": 3,  # Adjust gain to reduce noise
    "ExposureTime": 500,  # Increase exposure time for better lighting
    "Brightness": 0.0,  # Adjust brightness as needed
    "Contrast": 1,  # Increase contrast for better distinction
    "Saturation": 1,  # Increase saturation for more vibrant colors
    "Sharpness": 4,  # Increase sharpness for better detail
    "AeEnable": False,
    "AwbEnable": True
})
output = StreamingOutput()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    logging.info("Starting camera recording...")
    picam2.start_recording(MJPEGEncoder(bitrate=20000000), FileOutput(output))
    logging.info("Camera recording started.")
    
    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    logging.info("Starting server at http://localhost:8000")
    server.serve_forever()
except Exception as e:
    logging.error("Error occurred: %s", e)
finally:
    logging.info("Stopping camera recording...")
    picam2.stop_recording()
    logging.info("Camera recording stopped.")
