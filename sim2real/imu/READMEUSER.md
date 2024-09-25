# Python 数据读取IMU数据即模块配置例程 
# 先增加权限
打开所有的访问：
2.1 其中gedit用vim打开

sudo gedit /etc/udev/rules.d/70-ttyusb.rules

2.2 在该文件中添加如下一行（可能不存在此文件而创建一个新文件）

    KERNEL==“ttyUSB[0-9]*”, MODE=“0666”

2.3 重启系统即可

这样ttyUSB0-ttyUSB9默认的权限都变成了666，普通用户也可以读写串口了。

# 或者：
另一种是将该用户添加至dialout用户组，因为tty设备是属于dialout用户组，所以将用户添加到dialout用户组，该用户就具备了访问tty设备的权限；

3.1 查看串口信息

$ ls -l /dev/ttyUSB0
crw-rw---- 1 root dialout 4, 64 Jun  2 18:39 /dev/ttyUSB0

3.2 查看当前用户名

$ whoami

3.3 当前用户加入到dialout用户组

sudo usermod -aG dialout username

3.4 最后重启系统即可


# 调用imu的参数
直接运行imu.py即可看到terminal的输出

