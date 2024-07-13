

import safety_gymnasium

env_id = 'SafetyRacecarGoal0-v0'
render = False
if render:
    env = safety_gymnasium.make(env_id,render_mode="human")
else:
    env = safety_gymnasium.make(env_id)
obs, info = env.reset()
print("env ok.")
print(env.action_space)
exit()
while True:
    act = [0,0]
    obs, reward, cost, terminated, truncated, info = env.step(act)
    print(len(obs))
    print(obs)
    if terminated or truncated:
        break
    # env.render()

# Dict('accelerometer': Box(-inf, inf, (3,), float64),
# 'velocimeter': Box(-inf, inf, (3,), float64),
# 'gyro': Box(-inf, inf, (3,), float64),
# 'walls_lidar': Box(0.0, 1.0, (16,), float64),
# 'dis_x': Box(-inf, inf, (1,), float64), 'dis_y': Box(-inf, inf, (1,), float64)


# OBS
# accelerometer: 加速度计数据
# Type: Box(-inf, inf, (3,), float64)
# 说明：这是一个形状为(3, )的数组，包含三个浮点数值，用来表示在三个坐标轴上的加速度数据。通常是指物体在空间中的加速度，单位可能是米 / 秒²。

# velocimeter: 速度计数据
# Type: Box(-inf, inf, (3,), float64)
# 说明：同样是一个形状为(3, )的数组，包含三个浮点数值，表示物体在三个坐标轴上的速度数据。这些数据通常是由速度传感器或运动传感器提供的。

# gyro: 陀螺仪数据
# Type: Box(-inf, inf, (3,), float64)
# 说明：这是一个形状为(3, )的数组，包含三个浮点数值，表示物体绕三个坐标轴的角速度数据。陀螺仪通常用来测量旋转或姿态变化。

# magnetometer: 磁力计数据
# Type: Box(-inf, inf, (3,), float64)
# 说明：形状为(3, )的数组，包含三个浮点数值，用来表示物体在三个坐标轴上的磁场强度数据。磁力计通常用来检测地磁场或其他磁场变化。

# goal_lidar: 激光雷达数据
# Type: Box(0.0, 1.0, (16,), float64)
# 说明：形状为(16, )的数组，包含十六个浮点数值，每个值的范围在0.0到1.0之间。这通常是激光雷达传感器返回的数据，用来探测物体周围的环境，可能用于目标检测或避障。

# Action
# 后轮速度[-20,20]，前轮角度(rad)[0,1]