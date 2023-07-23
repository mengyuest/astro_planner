import gym
from gym import spaces
import numpy as np
from matplotlib.patches import Polygon, Rectangle, Ellipse, Circle
import matplotlib.pyplot as plt 
import torch
# G=6.674*(1e-11)

def heading_to_rotation_matrix(roll, pitch, yaw):
    # Convert heading angles (roll, pitch, yaw) to rotation matrix
    # Assuming roll-pitch-yaw Tait-Bryan convention
    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])

    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])

    # Combine the rotation matrices to get the final rotation matrix
    R_world_to_ego = R_yaw @ R_pitch @ R_roll
    return R_world_to_ego

def heading_to_rotation_matrix_torch(roll, pitch, yaw):
    # Convert heading angles (roll, pitch, yaw) to rotation matrix
    # Assuming roll-pitch-yaw Tait-Bryan convention
    R_roll = torch.tensor([[1, 0, 0],
                       [0, torch.cos(roll), -torch.sin(roll)],
                       [0, torch.sin(roll), torch.cos(roll)]])

    R_pitch = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)],
                        [0, 1, 0],
                        [-torch.sin(pitch), 0, torch.cos(pitch)]])

    R_yaw = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0],
                      [torch.sin(yaw), torch.cos(yaw), 0],
                      [0, 0, 1]])

    # Combine the rotation matrices to get the final rotation matrix
    R_world_to_ego = R_yaw @ R_pitch @ R_roll
    return R_world_to_ego

def transform_thrust_torque(thrust_ego, torque_ego, roll, pitch, yaw):
    # Convert thrust and torque from ego frame to world frame
    R_world_to_ego = heading_to_rotation_matrix_torch(roll, pitch, yaw)

    # Transform thrust vector
    thrust_world = R_world_to_ego @ thrust_ego

    # Transform torque vector
    torque_world = R_world_to_ego @ torque_ego

    return thrust_world, torque_world

class SpacecraftEnv(gym.Env):
    def __init__(self, args):
        self.args = args

        # state (planet_x, planet_y, planet_z, phi, ego_x, ego_y, ego_z, wx, wy, wz, vx, vy, vz, is_dropped, target_d, target_phi, x2, y2, z2, vx2, vy2, vz2)
        planet_dim = 4
        ship_dim = 12
        target_dim = 3 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(planet_dim + ship_dim + target_dim, ))
        # action (3d Thrust, 3d Torque and Drop action)
        self.action_space = spaces.Box(low=-10, high=10, shape=(3 + 3 + 1, ))
        self.state = np.zeros(self.observation_space.shape[0], )

        # coefficients
        # self.ship_mass = 1000.0
        # self.ship_Ixx = 1000.0
        # self.ship_Iyy = 1000.0
        # self.ship_Izz = 1000.0
        self.ship_mass = 1
        self.ship_Ixx = 1
        self.ship_Iyy = 1
        self.ship_Izz = 1

        #self.planet_mass = 1e5
        self.planet_mass = 1
        self.a = 10.0
        self.b = 5.0
        self.K = 0.1
        self.K1 = 1.0

        #self.G = 6.67430e-11
        self.G = 1e-7  # 100
        self.dt = 0.1
        self.time_limit = 1000

        # planet radius 
        self.planet_r = 1.0

        # planet state
        self.x0, self.y0, self.z0, self.phi0 = 0, 0, 0, 0

        # spaceship state
        self.x, self.y, self.z, self.roll, self.pitch, self.yaw = 0, 0, 0, 0, 0, 0
        self.vx, self.vy, self.vz, self.wx, self.wy, self.wz = 0, 0, 0, 0, 0, 0
        self.is_dropped, self.target_d, self.target_phi = 0, 0, 0

        self.make_observation()

        # for simulation statistics
        self.epi=0
        self.t=0

    def make_observation(self):
        self.observation = np.array([
            self.x0, self.y0, self.z0, self.phi0, self.x, self.y, self.z, self.roll, self.pitch, self.yaw,
            self.vx, self.vy, self.vz, self.wx, self.wy, self.wz, self.is_dropped, self.target_d, self.target_phi
            ])
    
    def unpack_observation(self, observation):
        self.x0, self.y0, self.z0, self.phi0, self.x, self.y, self.z, self.roll, self.pitch, self.yaw, \
            self.vx, self.vy, self.vz, self.wx, self.wy, self.wz, self.is_dropped, self.target_d, self.target_phi = observation.detach().cpu().numpy()

    def reset(self):
        # initialize planet state
        th = 0.0
        self.x0 = self.a * np.cos(self.K * th)
        self.y0 = self.b * np.sin(self.K * th)
        self.z0 = 0
        self.phi0 = np.pi/6

        # initialize spaceship state
        # ego_x = -3.5
        # ego_y = -5.2
        # theta = np.pi/6
        # v = 3.0

        self.x = -3.5
        self.y = -5.2
        self.z = 0

        # TODO (how to target)
        self.roll = 0
        self.pitch = 0
        self.yaw = np.arctan2(self.y0-self.y, self.x0-self.x)

        v = 3.0
        dx = self.x0 - self.x
        dy = self.y0 - self.y
        dz = self.z0 - self.z
        self.vx = v * dx / (dx**2+dy**2+dz**2)**0.5
        self.vy = v * dy / (dx**2+dy**2+dz**2)**0.5
        self.vz = v * dz / (dx**2+dy**2+dz**2)**0.5

        self.wx = 0
        self.wy = 0
        self.wz = 0

        # initialize target state
        self.is_dropped = 0
        self.target_d = 3
        self.target_phi = np.pi 
        
        self.make_observation()

        self.epi+=1
        self.t=0
        self.trajs=[]
    
    # physics dynamics
    def next_step(self, state, action, dt):
        x0, y0, z0, phi0, x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz, is_dropped, target_d, target_phi = state

        th = torch.atan2(y0 / self.b, x0 / self.a)
        th = (th + self.K * dt) 
        th = torch.atan2(torch.sin(th), torch.cos(th))

        new_x0 = self.a * torch.cos(th) 
        new_y0 = self.b * torch.sin(th)
        new_z0 = z0 * 0
        new_phi0 = phi0 + dt * self.K1

        thrust_ego = action[:3] * torch.tensor([1.0, 0.0, 0.0])
        torque_ego = action[3:6]

        thrust_vec, torque_vec = transform_thrust_torque(thrust_ego, torque_ego, roll, pitch, yaw)

        Fx, Fy, Fz = thrust_vec
        Tx, Ty, Tz = torque_vec

        r = torch.norm(state[0:3]-state[4:7])
        
        r_clip = torch.clip(r, 1e-1)

        Gx = self.G * self.planet_mass * self.ship_mass * (x0 - x) / (r_clip**3)
        Gy = self.G * self.planet_mass * self.ship_mass * (y0 - y) / (r_clip**3)
        Gz = self.G * self.planet_mass * self.ship_mass * (z0 - z) / (r_clip**3)

        ax = (Fx + Gx) / self.ship_mass
        ay = (Fy + Gy) / self.ship_mass
        az = (Fz + Gz) / self.ship_mass

        alpha_x = Tx / self.ship_Ixx
        alpha_y = Ty / self.ship_Iyy
        alpha_z = Tz / self.ship_Izz

        new_x = x + vx * dt
        new_y = y + vy * dt
        new_z = z + vz * dt

        new_roll = roll + wx * dt
        new_pitch = pitch + wy * dt
        new_yaw = yaw + wz * dt

        new_vx = vx + ax * dt
        new_vy = vy + ay * dt
        new_vz = vz + az * dt

        new_wx = wx + alpha_x * dt
        new_wy = wy + alpha_y * dt
        new_wz = wz + alpha_z * dt

        # print(new_wx, new_wy, new_wz, new_roll, new_pitch, new_yaw)

        new_roll = torch.atan2(torch.sin(new_roll), torch.cos(new_roll))
        new_pitch = torch.atan2(torch.sin(new_pitch), torch.cos(new_pitch))
        new_yaw = torch.atan2(torch.sin(new_yaw), torch.cos(new_yaw))

        new_state = torch.stack([
            new_x0, new_y0, new_z0, new_phi0, 
            new_x, new_y, new_z, new_roll, new_pitch, new_yaw, 
            new_vx, new_vy, new_vz, new_wx, new_wy, new_wz, 
            is_dropped, target_d, target_phi
        ])
        return new_state

    def step(self, action):
        self.trajs.append([self.x,self.y,self.z])

        new_state = self.next_step(torch.from_numpy(self.observation).float(), torch.from_numpy(action).float(), self.dt)

        # # planet dynamics
        # th = np.arctan2(self.y0/self.b, self.x0/self.a)
        # th = th + self.K * self.dt
        # self.x0 = self.a * np.cos(th)
        # self.y0 = self.b * np.sin(th)
        # self.z0 = 0
        # self.phi0 = self.phi0 + self.dt * self.K1

        # # spaceship dynamics
        # thrust_ego = action[:3]
        # torque_ego = action[3:6]
        # thrust_vec, torque_vec = transform_thrust_torque(thrust_ego, torque_ego, self.roll, self.pitch, self.yaw)
        # Fx, Fy, Fz = thrust_vec
        # Tx, Ty, Tz = torque_vec

        # # calculate gravitational force
        # r = np.linalg.norm([self.x-self.x0, self.y-self.y0, self.z-self.z0])
        # Gx = self.G * self.planet_mass * self.ship_mass * (self.x0 - self.x) / (r**3)
        # Gy = self.G * self.planet_mass * self.ship_mass * (self.y0 - self.y) / (r**3)
        # Gz = self.G * self.planet_mass * self.ship_mass * (self.z0 - self.z) / (r**3)

        # ax = (Fx + Gx) / self.ship_mass
        # ay = (Fy + Gy) / self.ship_mass
        # az = (Fz + Gz) / self.ship_mass

        # alpha_x = Tx / self.ship_Ixx
        # alpha_y = Ty / self.ship_Iyy
        # alpha_z = Tz / self.ship_Izz

        # new_x = self.x + self.vx * self.dt
        # new_y = self.y + self.vy * self.dt
        # new_z = self.z + self.vz * self.dt

        # new_roll = self.roll + self.wx * self.dt
        # new_pitch = self.pitch + self.wy * self.dt
        # new_yaw = self.yaw + self.wz * self.dt

        # new_vx = self.vx + ax * self.dt
        # new_vy = self.vy + ay * self.dt
        # new_vz = self.vz + az * self.dt

        # new_wx = self.wx + alpha_x * self.dt
        # new_wy = self.wy + alpha_y * self.dt
        # new_wz = self.wz + alpha_z * self.dt

        # new_roll = np.arctan2(np.sin(new_roll), np.cos(new_roll))
        # new_pitch = np.arctan2(np.sin(new_pitch), np.cos(new_pitch))
        # new_yaw = np.arctan2(np.sin(new_yaw), np.cos(new_yaw))

        # self.x, self.y, self.z, self.roll, self.pitch, self.yaw = new_x, new_y, new_z, new_pitch, new_roll, new_yaw
        # self.vx, self.vy, self.vz, self.wx, self.wy, self.wz = new_vx, new_vy, new_vz, new_wx, new_wy, new_wz
        # self.is_dropped, self.target_d, self.target_phi = self.is_dropped, self.target_d, self.target_phi
        
        self.unpack_observation(new_state)
        self.make_observation()

        # reward function and other status
        r = torch.norm(new_state[0:3]-new_state[4:7])
        dist_reward = -(r - self.target_d)**2 
        angle_reward = -abs((np.arctan2(self.y-self.y0, self.x-self.x0) - self.target_phi) % (np.pi*2) - np.pi)
        reward = dist_reward + angle_reward
        done = False
        info = {}

        self.t+=1

        return self.observation, reward, done, info
    
    def ship_polygon(self, x, y, z, roll, pitch, yaw):
        points = np.array([
            [1.25, 0],
            [0.5, 0.3],
            [-0.5, 0.3],
            [-1, 0.5],
            [-1, -0.5],
            [-0.5, -0.3],
            [0.5, -0.3],
        ])
        scale = 0.4
        R = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        points = (R @ (scale * points.T)).T + np.array([[x, y]])
        return points

    def render(self):
        plt.figure(figsize=(10, 10))
        # orbit
        ns = 1000
        ths = np.linspace(0, np.pi*2, ns)
        xs = self.a * np.cos(ths)
        ys = self.b * np.sin(ths)
        plt.plot(xs, ys, color='gray', alpha=0.5, linewidth=3.0)

        # planet
        circ=Circle([self.x0, self.y0], self.planet_r, color="royalblue", alpha=0.4)
        ax=plt.gca()
        ax.add_patch(circ)

        # phase direction
        x_from = self.x0
        x_to = self.x0 + self.planet_r * np.cos(self.phi0)
        y_from = self.y0
        y_to = self.y0 + self.planet_r * np.sin(self.phi0)
        plt.plot([x_from, x_to], [y_from, y_to], color="red")

        # spaceship
        pts = self.ship_polygon(self.x, self.y, self.z, self.roll, self.pitch, self.yaw)
        poly = Polygon(pts, color="brown", alpha=0.5)
        # circ=Circle([self.x, self.y], self.r, color="brown")

        x_from = self.x + 0.4 * np.cos(self.yaw+np.pi)
        x_to = self.x + 0.6 * np.cos(self.yaw+np.pi)
        y_from = self.y + 0.4 * np.sin(self.yaw+np.pi)
        y_to = self.y + 0.6 * np.sin(self.yaw+np.pi)
        plt.plot([x_from, x_to], [y_from, y_to], color="brown")

        # trajectory
        plt.plot([xx[0] for xx in self.trajs], [xx[1] for xx in self.trajs], color="brown", alpha=0.4, linestyle="-", markersize=4, marker='o')

        ax=plt.gca()
        ax.add_patch(poly)
        plt.axis("scaled")
        plt.xlim(-15, 15)
        plt.ylim(-13, 13)
        plt.savefig("sim_e%04d_t%04d.png"%(self.epi, self.t), bbox_inches='tight', pad_inches=0.1)
        plt.close()