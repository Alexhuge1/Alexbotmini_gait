import os
import struct
import array
from fcntl import ioctl
import threading
import time


class ps5_joystick:
    def __init__(self):
        self.axis_names = {
            0x00: 'abs_lx',  # 左摇杆X轴
            0x01: 'abs_ly',  # 左摇杆Y轴
            0x02: 'abs_l2',  # R2触发器
            0x05: 'abs_r2',  # L2触发器
            0x03: 'abs_rx',  # 右摇杆X轴
            0x04: 'abs_ry',  # 右摇杆Y轴
            0x10: 'corss_x',  # 十字X轴
            0x11: 'cross_y',  # 十字Y轴
        }
        self.button_names = {
            0x134: 'button_square',  # 正方形键（□按钮）
            0x130: 'button_cross',  # 交叉键（X按钮）
            0x131: 'button_circle',  # 圆形键（O按钮）
            0x13d: 'l_axis_button',  # 左摇杆按键
            0x13e: 'r_axis_button',  # 右摇杆按键
            0x136: 'button_l1',  # l1按钮
            0x137: 'button_r1',  # r1按钮
            0x13b: 'button_options',  # Options按钮
            0x13a: 'button_create',  # Create按钮
            0x13c: 'button_ps',  # PS按钮
        }
        self.axis_map = []
        self.button_map = []
        self.jsdev = None

        # 存储左右摇杆的值
        self.left_stick_x = 0
        self.left_stick_y = 0
        self.right_stick_x = 0
        self.right_stick_y = 0

        # 打开PS5手柄设备
        fn = '/dev/input/js0'
        print('Opening %s...' % fn)
        try:
            self.jsdev = open(fn, 'rb')
        except FileNotFoundError:
            print("please connect the joystick")
            import sys
            sys.exit(1)

        # Get the device name.
        buf = array.array('B', [0] * 64)
        ioctl(self.jsdev, 0x80006a13 + (0x10000 * len(buf)), buf)  # JSIOCGNAME(len)
        js_name = buf.tobytes().rstrip(b'\x00').decode('utf-8')
        print('Device name: %s' % js_name)

        # Get number of axes and buttons.
        buf = array.array('B', [0])
        ioctl(self.jsdev, 0x80016a11, buf)  # JSIOCGAXES
        num_axes = buf[0]

        buf = array.array('B', [0])
        ioctl(self.jsdev, 0x80016a12, buf)  # JSIOCGBUTTONS
        num_buttons = buf[0]

        # Get the axis map.
        buf = array.array('B', [0] * 0x40)
        ioctl(self.jsdev, 0x80406a32, buf)  # JSIOCGAXMAP

        for axis in buf[:num_axes]:
            axis_name = self.axis_names.get(axis, 'unknown(0x%02x)' % axis)
            self.axis_map.append(axis_name)

        # Get the button map.
        buf = array.array('H', [0] * 200)
        ioctl(self.jsdev, 0x80406a34, buf)  # JSIOCGBTNMAP

        for btn in buf[:num_buttons]:
            btn_name = self.button_names.get(btn, 'unknown(0x%03x)' % btn)
            self.button_map.append(btn_name)

        print('%d axes found: %s' % (num_axes, ', '.join(self.axis_map)))
        print('%d buttons found: %s' % (num_buttons, ', '.join(self.button_map)))

    def handle_events(self):
        while True:
            evbuf = self.jsdev.read(8)
            if evbuf:
                time, value, type, number = struct.unpack('IhBB', evbuf)

                if type & 0x01:  # button
                    # 跳过编号为 0x02 的按键
                    if number == 0x02:
                        continue

                    button = self.button_map[number]
                    if button:
                        self.button_names[button] = value

                if type & 0x02:  # axis
                    if number == 0x02:
                        continue

                    axis = self.axis_map[number]
                    if number == 0x04:
                        value = -value
                    if number == 0x01:
                        value = -value
                    if number == 0x05:
                        value = -value

                    # 过滤绝对值小于 5000 的值
                    if abs(value) < 2000:
                        continue

                    if number == 0x00:  # 左摇杆X轴
                        self.left_stick_x = value
                    elif number == 0x01:  # 左摇杆Y轴
                        self.left_stick_y = value
                    elif number == 0x03:  # 右摇杆X轴
                        self.right_stick_x = value
                    elif number == 0x04:  # 右摇杆Y轴
                        self.right_stick_y = value

    def get_stick_values(self):
        return {
            "left_stick_x": self.left_stick_x,
            "left_stick_y": self.left_stick_y,
            "right_stick_x": self.right_stick_x,
            "right_stick_y": self.right_stick_y
        }


if __name__ == "__main__":
    joystick = ps5_joystick()

    # 启动事件处理线程
    event_thread = threading.Thread(target=joystick.handle_events)
    event_thread.daemon = True
    event_thread.start()

    try:
        while True:
            time.sleep(0.1)
            stick_values = joystick.get_stick_values()
            print("Left Stick: X={}, Y={}; Right Stick: X={}, Y={}".format(
                stick_values["left_stick_x"],
                stick_values["left_stick_y"],
                stick_values["right_stick_x"],
                stick_values["right_stick_y"]
            ))
    except KeyboardInterrupt:
        print("Exiting...")
