import sys
import os
from commands.utils import check_python_version
from commands.cmd_list import cmd_list
from commands.read_data import cmd_read
from commands.cmd_send import cmd_send

if __name__ == "__main__":
    cmd_read('/dev/ttyUSB0', 115200)
