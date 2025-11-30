import gymnasium as gym
import numpy as np
import math
import pybullet as p
import os
import pybullet_data
import matplotlib.pyplot as plt

# Car class
class Car:
    def __init__(self, client):
        # Initialize vehicle
        self.client = client
        # Load vehicle model
        f_name = os.path.join(os.path.dirname(__file__), 'car.urdf')
        self.car = p.loadURDF(fileName=f_name,
                              basePosition=[-9, 0, 0.1],
                              physicsClientId=client)

        # Connect vehicle's joints
        joint_name_to_index = {}
        num_joints = p.getNumJoints(self.car)
        for i in range(num_joints):
            info = p.getJointInfo(self.car, i)
            name = info[1].decode('utf-8')
            joint_name_to_index[name] = i

        self.steering_joints = [
            joint_name_to_index.get('base_to_left_hinge'),
            joint_name_to_index.get('base_to_right_hinge')
        ]
        self.steering_joints = [j for j in self.steering_joints if j is not None]
        self.drive_joints = [
            joint_name_to_index.get('left_hinge_to_left_front_wheel'),
            joint_name_to_index.get('right_hinge_to_right_front_wheel'),
            joint_name_to_index.get('base_to_left_mid_wheel'),
            joint_name_to_index.get('base_to_right_mid_wheel'),
            joint_name_to_index.get('base_to_left_back_wheel'),
            joint_name_to_index.get('base_to_right_back_wheel')
        ]
        self.drive_joints = [j for j in self.drive_joints if j is not None]
        for link_idx in [-1]:
            p.changeDynamics(
                self.car,
                link_idx,
                lateralFriction=0.8,
                rollingFriction=0.0,
                spinningFriction=0.05,
                restitution=0.0,
                linearDamping=0.04,
                angularDamping=0.2,
                physicsClientId=self.client
            )
        for link_idx in self.drive_joints:
            p.changeDynamics(
                self.car,
                link_idx,
                lateralFriction=1.2,
                rollingFriction=0.02,
                spinningFriction=0.1,
                restitution=0.0,
                linearDamping=0.02,
                angularDamping=0.1,
                physicsClientId=self.client
            )
        # Joint speed
        self.joint_speed = 0
        # Rolling friction and air resistance
        self.c_rolling = 0.2
        self.c_drag = 0.01
        # Throttle constant
        self.c_throttle = 45
        self.max_joint_speed = 80  # 限制最大轮速
        # # 降低重心 + 增加质量 + 增加阻尼
        # p.changeDynamics(
        #     self.car,
        #     -1,  # base link
        #     mass=15.0,  # 增加质量（从7.22→15）提高稳定性
        #     localInertiaDiagonal=[0.15, 0.15, 0.08],  # 降低惯性矩，减少翻滚
        #     angularDamping=2.0,  # 角阻尼（防止翻滚）
        #     physicsClientId=client
        # )
        #

    def get_ids(self):
        return self.car, self.client

    def apply_action(self, action):
        # Accept action, which includes throttle and steering angle
        throttle, steering_angle = action

        # Limit throttle and steering angle range
        throttle = min(max(throttle, 0), 1)
        steering_angle = max(min(steering_angle, 0.6), -0.6)

        # Set the target angle for steering joints
        p.setJointMotorControlArray(self.car, self.steering_joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[steering_angle] * len(self.steering_joints),
                                    physicsClientId=self.client)

        # Calculate vehicle's friction and mechanical resistance
        friction = -self.joint_speed * (self.joint_speed * self.c_drag +
                                        self.c_rolling)
        acceleration = self.c_throttle * throttle + friction
        # Each time step is 1/30 seconds
        self.joint_speed = self.joint_speed + 1/30 * acceleration
        self.joint_speed = max(0, min(self.joint_speed, self.max_joint_speed))
        # if self.joint_speed < 0:
        #     self.joint_speed = 0

        # # Dynamic force calculation based on speed and throttle
        # if self.joint_speed < 10:
        #     # 启动阶段：需要大力克服静摩擦
        #     base_force = 50
        # elif self.joint_speed < 30:
        #     # 加速阶段：中等力
        #     base_force = 40
        # elif self.joint_speed < 60:
        #     # 巡航阶段：小力维持
        #     base_force = 25
        # else:
        #     # 高速阶段：最小力
        #     base_force = 20
        #
        # # 油门调制（保留部分基础力）
        # drive_force = base_force * (0.3 + throttle * 0.7)
        #
        # p.setJointMotorControlArray(
        #     bodyUniqueId=self.car,
        #     jointIndices=self.drive_joints,
        #     controlMode=p.VELOCITY_CONTROL,
        #     targetVelocities=[self.joint_speed] * len(self.drive_joints),
        #     forces=[drive_force] * len(self.drive_joints),
        #     physicsClientId=self.client)


        # Set target speed for wheels
        p.setJointMotorControlArray(
            bodyUniqueId=self.car,
            jointIndices=self.drive_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[self.joint_speed] * len(self.drive_joints),
            forces=[2.4]* len(self.drive_joints),
            physicsClientId=self.client)

    def get_observation(self):
        # Get the vehicle's position and orientation
        pos, ang = p.getBasePositionAndOrientation(self.car, self.client)
        ang = p.getEulerFromQuaternion(ang)
        
        # Extract orientation info
        roll = ang[0]
        pitch = ang[1]
        yaw = ang[2]
        
        # Original features
        ori = (math.cos(yaw), math.sin(yaw))
        pos_xy = pos[:2]  # x, y
        pos_z = (pos[2],) # z
        
        # Get the vehicle's velocity
        vel = p.getBaseVelocity(self.car, self.client)[0][0:2]
        
        # Combine: [x, y] + [z] + [cos, sin] + [pitch, roll] + [vx, vy]
        # Note: We group them to keep logical order, usually position -> orientation -> velocity
        observation = (pos_xy + pos_z + ori + (pitch, roll) + vel)
        
        return observation

# Goal class
class Goal:
    def __init__(self, client, base):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF(fileName="r2d2.urdf",
                   basePosition=[base[0], base[1], 0.2],
                   physicsClientId=client)

# Simple driving environment
class CarRobotEnv(gym.Env):
    def __init__(self):
        # Initialize environment
        super(CarRobotEnv, self).__init__()
        # Define action space: throttle and steering
        self.action_space = gym.spaces.Box(
            low=np.array([0, -.6], dtype=np.float32),
            high=np.array([1, .6], dtype=np.float32)
        )
        # Define observation space: vehicle's position, velocity, etc.
        # Modified: Added z, pitch, roll (3 extra dimensions) -> Total 11 dimensions
        # [x, y, z, cos(yaw), sin(yaw), pitch, roll, vx, vy, gx, gy]
        self.observation_space = gym.spaces.Box(
            low=np.array([-10, -10, -2, -1, -1, -3.14, -3.14, -5, -5, -10, -10], dtype=np.float32),
            high=np.array([10, 10, 5, 1, 1, 3.14, 3.14, 5, 5, 10, 10], dtype=np.float32)
        )
        # Set random number generator
        self.np_random, _ = gym.utils.seeding.np_random()

        # Connect to physics engine
        self.client = p.connect(p.GUI)
        
        # Set simulation time step
        p.setTimeStep(1/30, self.client)

        # Initialize car, goal, done state, etc.
        self.car = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.max_steps = 500  # 最大步数（可根据需要调整）
        self.current_step = 0  # 当前步数计数器
        # 地形类型设置：'heightfield' 或 'obstacles'
        self.terrain_type = 'heightfield'  # 默认使用高度场地形，可改为 'obstacles'
        # ========== 地形数据存储（用于计算坡度奖励） ==========
        self.terrain_data = None
        self.terrain_size = 20
        self.terrain_resolution = 100

        self.reset()  # Reset environment

    def create_heightfield_terrain(self):
        """创建高度场地形（起伏的山丘地形）"""
        terrain_size = 20  # 地形大小 20x20
        resolution = 100  # 分辨率
        # ========== 保存参数供后续使用 ==========
        self.terrain_size = terrain_size
        self.terrain_resolution = resolution
        # 生成高度数据（使用简单的正弦波叠加）
        height_data = np.zeros((resolution, resolution))
        for i in range(resolution):
            for j in range(resolution):
                x = i / resolution * 4 * np.pi
                y = j / resolution * 4 * np.pi
                # 叠加多个正弦波创建自然起伏
                height_data[i][j] = (np.sin(x) * np.cos(y) * 0.1 +
                                     np.sin(x * 2) * 0.15 +
                                     np.cos(y * 2) * 0.15)
        # ========== 保存地形数据用于计算坡度 ==========
        self.terrain_data = height_data
        # 创建高度场
        terrain_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[terrain_size / resolution, terrain_size / resolution, 1],
            heightfieldTextureScaling=(resolution - 1) / 2,
            heightfieldData=height_data.flatten(),
            numHeightfieldRows=resolution,
            numHeightfieldColumns=resolution,
            physicsClientId=self.client
        )

        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=terrain_shape,
            basePosition=[0, 0, 0],
            physicsClientId=self.client
        )

        # 修改地形颜色（可选，让地形更明显）
        p.changeVisualShape(terrain_shape, -1, rgbaColor=[0.4, 0.6, 0.3, 1],
                            physicsClientId=self.client)

    def create_box_obstacle(self, position, size, color=[0.5, 0.3, 0.1, 1]):
        """创建盒子形状的障碍物"""
        half_extents = [size[0] / 2, size[1] / 2, size[2] / 2]

        box_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.client
        )
        box_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            physicsClientId=self.client
        )
        p.createMultiBody(
            baseMass=0,  # 质量为0表示静态物体
            baseCollisionShapeIndex=box_collision,
            baseVisualShapeIndex=box_visual,
            basePosition=[position[0], position[1], position[2] + half_extents[2]],
            physicsClientId=self.client
        )

    def create_cylinder_obstacle(self, position, radius, height, color=[0.3, 0.5, 0.7, 1]):
        """创建圆柱形障碍物"""
        cylinder_collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=height,
            physicsClientId=self.client
        )
        cylinder_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=color,
            physicsClientId=self.client
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=cylinder_collision,
            baseVisualShapeIndex=cylinder_visual,
            basePosition=[position[0], position[1], position[2] + height / 2],
            physicsClientId=self.client
        )

    def create_ramp(self, position, size, angle, color=[0.4, 0.4, 0.4, 1]):
        """创建斜坡"""
        angle_rad = math.radians(angle)
        half_extents = [size[0] / 2, size[1] / 2, size[2] / 2]

        ramp_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.client
        )
        ramp_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            physicsClientId=self.client
        )

        # 计算斜坡的旋转和位置
        orientation = p.getQuaternionFromEuler([angle_rad, 0, 0])
        adjusted_position = [
            position[0],
            position[1],
            position[2] + half_extents[2] * math.cos(angle_rad) + half_extents[0] * math.sin(angle_rad)
        ]

        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=ramp_collision,
            baseVisualShapeIndex=ramp_visual,
            basePosition=adjusted_position,
            baseOrientation=orientation,
            physicsClientId=self.client
        )

    def create_obstacle_terrain(self):
        """创建带有多种障碍物的地形"""
        # 基础平面
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF(fileName="plane.urdf")

        # 添加固定位置的障碍物（保证一致性）
        # 区域1: 左下角的盒子障碍物群
        self.create_box_obstacle(position=[-5, -5, 0], size=[1.5, 1.5, 0.8], color=[0.7, 0.3, 0.2, 1])
        self.create_box_obstacle(position=[-3, -6, 0], size=[1.0, 2.0, 0.6], color=[0.6, 0.4, 0.2, 1])
        self.create_box_obstacle(position=[-6, -3, 0], size=[2.0, 1.0, 0.5], color=[0.5, 0.3, 0.3, 1])

        # 区域2: 右上角的圆柱障碍物
        self.create_cylinder_obstacle(position=[5, 5, 0], radius=0.8, height=1.0, color=[0.3, 0.6, 0.8, 1])
        self.create_cylinder_obstacle(position=[7, 6, 0], radius=0.6, height=0.8, color=[0.4, 0.5, 0.7, 1])
        self.create_cylinder_obstacle(position=[6, 3, 0], radius=0.5, height=1.2, color=[0.2, 0.5, 0.9, 1])

        # 区域3: 中间区域的混合障碍物
        self.create_box_obstacle(position=[0, -2, 0], size=[2.5, 1.0, 0.4], color=[0.8, 0.6, 0.2, 1])
        self.create_cylinder_obstacle(position=[2, 2, 0], radius=0.7, height=0.9, color=[0.5, 0.3, 0.6, 1])
        self.create_box_obstacle(position=[-2, 4, 0], size=[1.2, 1.2, 0.7], color=[0.3, 0.7, 0.3, 1])

        # 添加一些小型障碍物（增加复杂度）
        small_obstacles = [
            ([-7, 2, 0], [0.6, 0.6, 0.3]),
            ([8, -4, 0], [0.7, 0.7, 0.4]),
            ([1, 7, 0], [0.5, 0.5, 0.5]),
            ([-1, -8, 0], [0.8, 0.5, 0.3]),
            ([4, 8, 0], [0.6, 0.8, 0.4])
        ]

        for pos, size in small_obstacles:
            self.create_box_obstacle(
                position=pos,
                size=size,
                color=[np.random.uniform(0.3, 0.8),
                       np.random.uniform(0.3, 0.8),
                       np.random.uniform(0.2, 0.7),
                       1]
            )

    def get_terrain_slope(self, x, y):
        """获取指定位置的地形坡度（仅对高度场地形有效）"""
        if self.terrain_data is None:
            return 0.0  # 障碍物地形返回0

        # 将世界坐标转换为地形数组索引
        # 地形中心在(0,0)，范围是[-10, 10]
        i = int((x + self.terrain_size / 2) / self.terrain_size * self.terrain_resolution)
        j = int((y + self.terrain_size / 2) / self.terrain_size * self.terrain_resolution)

        # 边界检查
        if i < 1 or i >= self.terrain_resolution - 1 or j < 1 or j >= self.terrain_resolution - 1:
            return 0.0

        # 计算局部坡度（使用中心差分）
        dx = (self.terrain_data[i + 1, j] - self.terrain_data[i - 1, j]) / 2
        dy = (self.terrain_data[i, j + 1] - self.terrain_data[i, j - 1]) / 2

        # 坡度大小（梯度的模）
        slope = math.sqrt(dx ** 2 + dy ** 2)

        return slope

    def step(self, action):
        # Apply action to the car and update its state
        self.car.apply_action(action)
        p.stepSimulation()  # Execute a physics simulation step
        car_ob = self.car.get_observation()  # Get car's current observation

        # 增加步数计数
        self.current_step += 1

        # Calculate reward using the calculate_reward function
        reward, dist_to_goal = self.calculate_reward(car_ob)
        self.prev_dist_to_goal = dist_to_goal  # Update previous distance

        # Print the reward
        print(f"Reward: {reward}")

        # Assume no time limit, so `truncated` is set to False
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
            print(f"Episode truncated: reached max steps ({self.max_steps})")

        # Combine vehicle observation and goal location as new observation
        ob = np.array(car_ob + self.goal, dtype=np.float32)

        # Return the necessary five values: observation, reward, done, truncated, info
        info = {}  # Additional information can be expanded if needed
        return ob, reward, self.done, truncated, info

    def calculate_reward(self, car_ob):
        # Calculate distance to the goal
        dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                (car_ob[1] - self.goal[1]) ** 2))
        
        # Calculate reward based on distance
        reward = abs(self.prev_dist_to_goal - dist_to_goal)  # Reward based on distance

        # 地形平缓度奖励（仅对高度场地形）
        if self.terrain_data is not None:
            slope = self.get_terrain_slope(car_ob[0], car_ob[1])
            # 坡度越小，奖励越大（平缓路面奖励）
            # 使用指数衰减：坡度为0时奖励最大(0.1)，坡度越大奖励越小
            slope_reward = 0.5 * math.exp(-slope * 5)
            reward += slope_reward

        # Check if done: whether the car is out of bounds
        if (car_ob[0] >= 10 or car_ob[0] <= -10 or
            car_ob[1] >= 10 or car_ob[1] <= -10):
            self.done = True
            reward = 0  # Penalize the car if it goes out of bounds

        # Check if done: whether the car reaches the goal
        elif dist_to_goal < 1:
            self.done = True
            reward = 10  # High reward if the car reaches the goal
            print(f"Goal reached! Reward: {reward}")

        # Return the calculated reward
        return reward, dist_to_goal

    def seed(self, seed=None):
        # Set random seed
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def draw_boundary(self):
        # Set the size of the boundary
        x_range = 10  # x-axis range
        y_range = 10  # y-axis range
        z_height = 1  # Line height, set to 1 to show lines above the ground

        # Draw the boundary
        p.addUserDebugLine([x_range, y_range, z_height], [x_range, -y_range, z_height], lineColorRGB=[1, 0, 0], physicsClientId=self.client)
        p.addUserDebugLine([x_range, -y_range, z_height], [-x_range, -y_range, z_height], lineColorRGB=[1, 0, 0], physicsClientId=self.client)
        p.addUserDebugLine([-x_range, -y_range, z_height], [-x_range, y_range, z_height], lineColorRGB=[1, 0, 0], physicsClientId=self.client)
        p.addUserDebugLine([-x_range, y_range, z_height], [x_range, y_range, z_height], lineColorRGB=[1, 0, 0], physicsClientId=self.client)

    def reset(self, seed=None, options=None):
        # Reset simulation
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8)  # Set gravity
        self.current_step = 0
        # Reload plane and car
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # p.loadURDF(fileName="plane.urdf")
        if options and 'terrain_type' in options:
            terrain_type = options['terrain_type']
        else:
            terrain_type = self.terrain_type

        if terrain_type == 'heightfield':
            self.create_heightfield_terrain()  # 高度场地形
        else:
            self.create_obstacle_terrain()  # 障碍物地形
        self.car = Car(self.client)
        
        # Create boundaries
        self.draw_boundary()

        # Randomly set the goal position
        x = self.np_random.choice([i for i in range(0, 10)])
        y = self.np_random.choice([i for i in range(-9, 10)])

        self.goal = (x, y)
        self.done = False

        # Display the goal position in the environment
        Goal(self.client, self.goal)

        # Get the initial observation of the car
        car_ob = self.car.get_observation()

        # Calculate the distance from the car to the goal
        self.prev_dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                             (car_ob[1] - self.goal[1]) ** 2))

        # Return observation and other info
        obs = np.array(car_ob + self.goal, dtype=np.float32)
        info = {}  # Additional information
        return obs, info

    def close(self):
        # Close connection to the PyBullet simulation client
        p.disconnect(self.client)
