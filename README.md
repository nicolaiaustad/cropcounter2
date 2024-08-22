# CropCounter

CropCounter is a project that automates image capture and GPS data processing using Raspberry Pi and a connected camera. The project is designed to run automatically when the system starts, leveraging udev rules and systemd services.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [System Requirements](#system-requirements)
- [Project Structure](#project-structure)
- [Contributing](#contributing)


## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd CropCounter

Run the .config/setup.sh file to install packages and dependencies
Run the .config/setup_configs.sh file to setup udev rules and systemd services

Remember to update the idVendor and idProduct numbers for the usb stick you use as a switch in the udev rule file 99-usb-stick.rules
