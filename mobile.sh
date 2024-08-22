#!/bin/bash

SCRIPT_PATH="/home/nicolaiaustad/Desktop/CropCounter/run.py"
LOG_FILE="/home/nicolaiaustad/Desktop/CropCounter/run.log"

start() {
    echo "Starting the script..."
    nohup python3 $SCRIPT_PATH 2>&1 | tee -a $LOG_FILE &
    echo "Script started with PID $!"
}

stop() {
    echo "Stopping the script..."
    PID=$(ps aux | grep $SCRIPT_PATH | grep -v grep | awk '{print $2}')
    if [ -z "$PID" ]; then
        echo "Script is not running."
    else
        kill $PID
        echo "Script stopped."
    fi
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        start
        ;;
    status)
        PID=$(ps aux | grep $SCRIPT_PATH | grep -v grep | awk '{print $2}')
        if [ -z "$PID" ]; then
            echo "Script is not running."
        else
            echo "Script is running with PID $PID."
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
esac
