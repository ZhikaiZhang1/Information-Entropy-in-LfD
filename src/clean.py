with open('C:/Users/zhika/Downloads/eigenbot_dev_ws/gait_percent_joint_controller.py', 'rb+') as f:
    content = f.read()
    f.seek(0)
    f.write(content.replace(b'\r', b''))
    f.truncate()