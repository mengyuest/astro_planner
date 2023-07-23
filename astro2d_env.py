import gym
from gym import spaces
import numpy as np
from matplotlib.patches import Polygon, Rectangle, Ellipse, Circle
import matplotlib.pyplot as plt 
import torch
# G=6.674*(1e-11

class SpacecraftEnv2D(gym.Env):
    def __init__(self, args):
        self.args = args

        # state (planet_x, planet_y, phi, ego_x, ego_y, yaw, vx, vy, wz, is_dropped, target_d, target_phi)
        planet_dim = 3
        ship_dim = 6
        target_dim = 2 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(planet_dim + ship_dim + target_dim, ))
        # action (1d Thrust, 1d Torque and Drop action)
        self.action_space = spaces.Box(low=-10, high=10, shape=(1 + 1 + 1, ))
        self.state = np.zeros(self.observation_space.shape[0], )

        # coefficients
        self.ship_mass = 1
        self.ship_Ixx = 1
        self.ship_Iyy = 1
        self.ship_Izz = 1

        #self.planet_mass = 1e5
        self.planet_mass = 1
        self.a = 10.0
        self.b = 5.0
        self.K = 0.05
        self.K1 = 0.3

        #self.G = 6.67430e-11
        self.G = 0.0  # 100
        self.dt = 0.25
        self.time_limit = 1000

        # planet radius 
        self.planet_r = 4.0

        # planet state
        self.x0, self.y0, self.phi0 = 0, 0, 0

        # spaceship state
        self.x, self.y, self.yaw = 0, 0, 0
        self.vx, self.vy, self.wz = 0, 0, 0
        self.is_dropped, self.target_d, self.target_phi = 0, 0, 0
        self.make_observation()

        # for simulation statistics
        self.epi=0
        self.t=0

    def make_observation(self):
        self.observation = np.array([
            self.x0, self.y0, self.phi0, self.x, self.y, self.yaw,
            self.vx, self.vy, self.wz, self.is_dropped, self.target_d, self.target_phi
            ])
    
    def unpack_observation(self, observation):
        self.x0, self.y0, self.phi0, self.x, self.y, self.yaw,\
            self.vx, self.vy, self.wz, self.is_dropped, self.target_d, self.target_phi = observation.detach().cpu().numpy()

    def reset(self):
        # initialize planet state
        th = 0.0
        self.x0 = self.a * np.cos(self.K * th)
        self.y0 = self.b * np.sin(self.K * th)
        self.phi0 = np.pi/6 + np.pi

        # initialize spaceship state
        self.x = -3.5
        self.y = -5.2
        self.z = 0

        # TODO (how to target)
        self.yaw = np.arctan2(self.y0-self.y, self.x0-self.x)

        v = 3.0
        dx = self.x0 - self.x
        dy = self.y0 - self.y
        self.vx = v * dx / (dx**2+dy**2)**0.5
        self.vy = v * dy / (dx**2+dy**2)**0.5
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
        # (N, k), (N, d)
        # (N, k)
        WZ_MIN=-1
        WZ_MAX=1
        x0, y0, phi0, x, y, yaw, vx, vy, wz, is_dropped, target_d, target_phi = torch.split(state, split_size_or_sections=1, dim=-1)

        th = torch.atan2(y0 / self.b, x0 / self.a)
        th = (th + self.K * dt)
        th = torch.atan2(torch.sin(th), torch.cos(th))

        new_x0 = self.a * torch.cos(th) 
        new_y0 = self.b * torch.sin(th)
        new_phi0 = phi0 + dt * self.K1

        thrust_ego = action[:, 0:1]
        torque_ego = action[:, 1:2]
        Fx = thrust_ego * torch.cos(yaw)
        Fy = thrust_ego * torch.sin(yaw)
        Tz = torque_ego

        r = torch.norm(state[:, 0:2]-state[:, 3:5], dim=-1, keepdim=True)
        
        r_clip = torch.clip(r, 1e-1)

        Gx = self.G * self.planet_mass * self.ship_mass * (x0 - x) / (r_clip**3)
        Gy = self.G * self.planet_mass * self.ship_mass * (y0 - y) / (r_clip**3)

        ax = (Fx + Gx) / self.ship_mass
        ay = (Fy + Gy) / self.ship_mass
        alpha_z = Tz / self.ship_Izz

        new_x = x + vx * dt
        new_y = y + vy * dt
        new_yaw = yaw + wz * dt

        new_vx = vx + ax * dt
        new_vy = vy + ay * dt
        new_wz = torch.clip(wz + alpha_z * dt, WZ_MIN, WZ_MAX)

        new_yaw = torch.atan2(torch.sin(new_yaw), torch.cos(new_yaw))
        
        new_state = torch.cat([
            new_x0, new_y0, new_phi0, 
            new_x, new_y, new_yaw, 
            new_vx, new_vy, new_wz, 
            is_dropped, target_d, target_phi
        ], dim=-1)
        return new_state

    def step(self, action):
        self.trajs.append([self.x,self.y])

        new_state = self.next_step(torch.from_numpy(self.observation).float().unsqueeze(0), torch.from_numpy(action).float().unsqueeze(0), self.dt)

        self.unpack_observation(new_state[0])
        self.make_observation()

        # reward function and other status
        r = torch.norm(new_state[0:2]-new_state[3:5])
        dist_reward = -(r - self.target_d)**2 
        angle_reward = -abs((np.arctan2(self.y-self.y0, self.x-self.x0) - self.target_phi) % (np.pi*2) - np.pi)
        reward = dist_reward + angle_reward
        done = False
        info = {}

        self.t+=1

        return self.observation, reward, done, info
    
    def ship_polygon(self, x, y, yaw):
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
        pts = self.ship_polygon(self.x, self.y, self.yaw)
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
        filename = "sim_e%04d_t%04d.png"%(self.epi, self.t)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        return filename