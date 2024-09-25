import sys
sys.dont_write_bytecode = True
import time
import serial
from commands.utils import clear_screen
from commands.hipnuc_serial_parser import hipnuc_parser
from commands.hipnuc_nmea_parser import hipnuc_nmea_parser

def cmd_read(port, baudrate):

    serial_parser = hipnuc_parser()
    nmea_parser = hipnuc_nmea_parser()

    frame_count = 0
    frame_rate = 0
    last_frame_time = time.time()
    last_display_time = time.time()
    display_interval = 0.2  # Update display every 0.2 seconds

    latest_hipnuc_frame = None
    latest_nmea_frames = []

    try:
        with serial.Serial(port, int(baudrate), timeout=1) as ser:
            while True:
                if ser.in_waiting:
                    data = ser.read(ser.in_waiting)
                    try:
                        hipnuc_frames = serial_parser.parse(data)
                        nmea_frames = nmea_parser.parse(data.decode('ascii', errors='ignore'))
                        
                        frame_count += len(hipnuc_frames) + len(nmea_frames)
                        
                        if hipnuc_frames:
                            latest_hipnuc_frame = hipnuc_frames[-1]
                        if nmea_frames:
                            latest_nmea_frames = nmea_frames
                        
                        current_time = time.time()
                        if current_time - last_frame_time >= 1.0:
                            frame_rate = frame_count
                            frame_count = 0
                            last_frame_time = current_time

                        # Update display at fixed interval
                        if current_time - last_display_time >= display_interval:
                            clear_screen()
                            
                            if latest_hipnuc_frame:
                                serial_parser.print_parsed_data(latest_hipnuc_frame)
                            if latest_nmea_frames:
                                nmea_parser.print_parsed_data(latest_nmea_frames)
                            
                            print(f"Frame rate: {frame_rate} Hz")
                            last_display_time = current_time
                            
                    except Exception as e:
                        print(f"Error parsing data: {e}")

                time.sleep(0.001)  # Small delay to prevent CPU overuse

    except KeyboardInterrupt:
        print("Program interrupted by user")

if __name__ == "__rmain__":
    cmd_read()
