
import time
import random
import numpy as np
import torch
import imageio
import gym
from gym import spaces
from matplotlib.patches import Polygon, Rectangle, Ellipse, Circle
import matplotlib.pyplot as plt 

CHARGE_X = 10
CHARGE_Y = -2
class SpaceEnv(gym.Env):
    def __init__(self, args):
        planet_dim = 3
        ship_dim = 3  # (x, y, phi)
        target_dim = 3 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(planet_dim + ship_dim + target_dim, ))
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

        # planet radius 
        self.planet_r = 3.0

        # for simulation statistics
        self.epi=0
        self.t=0
        self.time_limit=100
    
    def next_step(self, state, action):
        # state representation
        # (x0, y0, phi0) position and rotation phase of the planet
        # (x, y, yaw) position and heading of the spaceship
        # (mode, x1, y1) the mode and coordinate for the sensor
        x0, y0, phi0, x, y, yaw, mode, x1, y1 = torch.split(state, split_size_or_sections=1, dim=-1)

        info = {"landed": False, "charged": False, "closed_enough": False}
        
        dt = self.dt

        th = torch.atan2(y0 / self.b, x0 / self.a)
        th = (th + self.K * dt)
        th = torch.atan2(torch.sin(th), torch.cos(th))
        
        new_x0 = self.a * torch.cos(th) 
        new_y0 = self.b * torch.sin(th)
        new_phi0 = phi0 + dt * self.K1

        v = action[:, 0:1]
        w = action[:, 1:2]
        mode = torch.clip(mode + action, 0, 1)
        
        new_x = x + v * torch.cos(yaw) * self.dt
        new_y = y + v * torch.sin(yaw) * self.dt
        new_yaw = yaw + w * self.dt

        on_surface = (((x1-x0)**2 + (y1-y0)**2)**0.5 <= self.planet_r).float()

        # surface movement
        # x10 = (x1 - x0) * np.cos(self.K * dt) - (y1 - y0) * np.sin(self.K * dt) + x0
        # y10 = (x1 - x0) * np.sin(self.K * dt) + (y1 - y0) * np.cos(self.K * dt) + y0
        th_ground = torch.atan2(y1-y0, x1-x0)
        new_th_ground = th_ground + dt * self.K1
        x10 = new_x0 + self.planet_r * torch.cos(new_th_ground)
        y10 = new_y0 + self.planet_r * torch.sin(new_th_ground)
        x11 = -(x1-x0) * 0.2 + x1
        y11 = -(y1-y0) * 0.2 + y1

        on_ground = (((x11-x0)**2 + (y11-y0)**2)**0.5 <= self.planet_r).float()
        th_est = torch.atan2(y11-y0, x11-x0)
        x11_ground = x0 + self.planet_r * torch.cos(th_est)
        y11_ground = y0 + self.planet_r * torch.sin(th_est)
        x11 = on_ground * x11_ground + (1-on_ground) * x11
        y11 = on_ground * y11_ground + (1-on_ground) * y11

        new_x1 = on_surface * x10 + (1-on_surface) * x11
        new_y1 = on_surface * y10 + (1-on_surface) * y11

        new_x1 = (mode==0).float() * new_x + (mode!=0).float() * new_x1
        new_y1 = (mode==0).float() * new_y + (mode!=0).float() * new_y1

        new_state = torch.cat([
            new_x0, new_y0, new_phi0, 
            new_x, new_y, new_yaw, 
            mode, new_x1, new_y1
        ], dim=-1)

        info["closed_enough"] = ((new_x - new_x0)**2 + (new_y - new_y0)**2)**0.5<self.planet_r + 2.0
        info["landed"] = on_ground[0]==1
        info["charged"] = ((new_x - CHARGE_X)**2 + (new_y - CHARGE_Y)**2)**0.5<1.0

        return new_state, info

    def reset(self):
        th = 0.0
        x0 = self.a * np.cos(self.K * th)
        y0 = self.b * np.cos(self.K * th)
        phi0 = np.pi/6 + np.pi

        # x = -3.5
        # y = -5.2
        x = -10
        y = 5.2
        z = 0

        yaw = np.arctan2(y0-y, x0-x)
        v = 3.0
        dx = x0 - x
        dy = y0 - y
        vx = v * dx / (dx**2+dy**2)**0.5
        vy = v * dy / (dx**2+dy**2)**0.5
        wz = 0

        mode = 0
        x1 = 0
        y1 = 0
        
        self.epi += 1
        self.t = 0
        self.trajs = []
        self.state = np.array([
            x0, y0, phi0, x, y, yaw, mode, x1, y1
        ]).astype(np.float32)

        return self.state

    def step(self, action):
        self.trajs.append(self.state)
        new_state, info = self.next_step(torch.from_numpy(self.state).float().unsqueeze(0), torch.from_numpy(action).float().unsqueeze(0))
        new_state = new_state[0]
        
        # change mode (0-reach certain point on the orbit; 1-circle the orbit; 2-pick up the sensor)
        # mode = new_state[-3]
        # close_thres = 0.3
        # if mode == 0:
        #     tgt_point = 
        #     # if close to the target point, mode -> 1
        #     if torch.norm(tgt_point-new_state[3:5], dim=-1) < close_thres:
        #         mode = 1
        # elif mode == 1:
        #     # if close to the target point, mode -> 2
        #     if torch.norm(tgt_point-new_state[3:5], dim=-1) < close_thres:
        #         mode = 2
        # elif mode == 2:
        #     # if have picked up the sensor, mode -> 3
        #     if torch.norm(new_state[3:5] - new_state[-2:], dim=-1) < close_thres:
        #         mode = 3
        # new_state[-3] = mode
        self.state = new_state.detach().cpu().numpy()
        
        # design reward
        reward = -torch.norm(new_state[0:2] - new_state[3:5])
        done = self.t>=self.time_limit
        self.t+=1
        return self.state, reward, done, info
    
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
    
    def sensor_polygon(self, x, y):
        n_angles = 10
        th_list = []
        for i in range(n_angles):
            th_list.append([np.cos(i/n_angles*np.pi*2), np.sin(i/n_angles*np.pi*2)])
        points = np.array(th_list)
        
        scale = 0.4
        # R = np.array([
        #     [np.cos(yaw), -np.sin(yaw)],
        #     [np.sin(yaw), np.cos(yaw)]
        # ])
        # points = (R @ (scale * points.T)).T + np.array([[x, y]])
        points = (scale * points.T).T + np.array([[x, y]])
        print(points)
        return points

    def render(self):
        plt.figure(figsize=(10, 10))
        # orbit
        ns = 1000
        ths = np.linspace(0, np.pi*2, ns)
        xs = self.a * np.cos(ths)
        ys = self.b * np.sin(ths)
        plt.plot(xs, ys, color='gray', alpha=0.5, linewidth=3.0)

        x0, y0, phi0, x, y, yaw, mode, x1, y1 = self.state

        # planet
        circ=Circle([x0, y0], self.planet_r, color="royalblue", alpha=0.4)
        ax=plt.gca()
        ax.add_patch(circ)

        # phase direction
        x_from = x0
        x_to = x0 + self.planet_r * np.cos(phi0)
        y_from = y0
        y_to = y0 + self.planet_r * np.sin(phi0)
        plt.plot([x_from, x_to], [y_from, y_to], color="red")

        # spaceship
        pts = self.ship_polygon(x, y, yaw)
        poly = Polygon(pts, color="brown", alpha=0.5)

        x_from = x + 0.4 * np.cos(yaw+np.pi)
        x_to = x + 0.6 * np.cos(yaw+np.pi)
        y_from = y + 0.4 * np.sin(yaw+np.pi)
        y_to = y + 0.6 * np.sin(yaw+np.pi)
        plt.plot([x_from, x_to], [y_from, y_to], color="brown")
        # trajectory
        plt.plot([xx[3] for xx in self.trajs], [xx[4] for xx in self.trajs], color="brown", alpha=0.4, linestyle="-", markersize=4, marker='o')
        ax=plt.gca()
        ax.add_patch(poly)

        # sensor
        if mode==1:
            pts1 = self.sensor_polygon(x1, y1)
            poly1 = Polygon(pts1, color="black", alpha=0.5)
            ax.add_patch(poly1)
        
        plt.axis("scaled")
        plt.xlim(-15, 15)
        plt.ylim(-13, 13)
        filename = "sim_e%04d_t%04d.png"%(self.epi, self.t)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        return filename

def gradient_planner(env, nt, ):
    num_iters = 100
    lr = 0.1

    # N = 20
    N = 1
    target_phi = torch.linspace(0, np.pi * 2, N)
    # controls = torch.zeros((N, nt, 3))
    
    controls = torch.zeros((N, nt, 3))
    controls.requires_grad = True
    optimizer = torch.optim.Adam([controls], lr=lr)
    for epi in range(num_iters):
        state = torch.from_numpy(env.state).float()
        state = torch.tile(state.unsqueeze(0), (N, 1))  # copy the state N times
        trajs = [state]
        for ti in range(nt):
            state, _ = env.next_step(state, controls[:, ti])
            trajs.append(state)
        trajs = torch.stack(trajs, dim=1)  # (N, T+1, K)

        target_radius = env.planet_r + 2.0
        target_phase = target_phi.reshape((N, 1))
        target_point_x = trajs[:, :, 0] + target_radius * torch.cos(target_phase+trajs[:, :, 2])
        target_point_y = trajs[:, :, 1] + target_radius * torch.sin(target_phase+trajs[:, :, 2])
        target_point = torch.stack([target_point_x, target_point_y], dim=-1)

        point_loss = torch.norm(target_point[:, -5:, 0:2]-trajs[:, -5:, 3:5], dim=-1)
        # dist = torch.norm(trajs[:, -5:, 0:2] - trajs[:, -5:, 3:5], dim=-1)
        # # safe_loss = torch.nn.ReLU()(env.planet_r + 0.01 - dist) * 10
        norm_loss = torch.mean(torch.norm(controls, dim=-1), dim=-1) * 0.01
        loss_vec = torch.mean(point_loss, dim=-1) + norm_loss
        loss = torch.mean(loss_vec)

        # dist = torch.norm(trajs[:, -5:, 3:5], dim=-1)
        # loss_vec = torch.mean(dist, dim=-1)
        # loss = torch.mean(loss_vec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epi, loss.item())

    min_idx = torch.argmin(loss_vec)
    print("idx=", min_idx, "loss=",loss_vec[min_idx])
    return controls[min_idx]

def gradient_planner_v1(env, nt, ):
    num_iters = 100
    lr = 0.1

    # N = 20
    # target_phi = torch.linspace(0, np.pi * 2, N)
    # controls = torch.zeros((N, nt, 3))
    N = 1
    controls = torch.zeros((N, nt, 3))
    controls.requires_grad = True
    optimizer = torch.optim.Adam([controls], lr=lr)
    charge_dest = torch.zeros(N, nt, 2)
    charge_dest[:, :, 0] = CHARGE_X
    charge_dest[:, :, 1] = CHARGE_Y
    for epi in range(num_iters):
        state = torch.from_numpy(env.state).float()
        state = torch.tile(state.unsqueeze(0), (N, 1))  # copy the state N times
        trajs = [state]
        for ti in range(nt):
            state, _ = env.next_step(state, controls[:, ti])
            trajs.append(state)
        trajs = torch.stack(trajs, dim=1)  # (N, T+1, K)
        dist = torch.norm(trajs[:, -5:, 3:5]-charge_dest[:, -5:, :], dim=-1)
        loss_vec = torch.mean(dist, dim=-1)
        loss = torch.mean(loss_vec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epi, loss.item())

    min_idx = torch.argmin(loss_vec)
    print("idx=", min_idx, "loss=",loss_vec[min_idx])
    return controls[min_idx]

def gradient_planner_v2(env, nt, ):
    num_iters = 100
    lr = 0.1

    # N = 20
    # target_phi = torch.linspace(0, np.pi * 2, N)
    # controls = torch.zeros((N, nt, 3))
    N = 1
    controls = torch.zeros((N, nt, 3))
    controls.requires_grad = True
    optimizer = torch.optim.Adam([controls], lr=lr)
    for epi in range(num_iters):
        state = torch.from_numpy(env.state).float()
        state = torch.tile(state.unsqueeze(0), (N, 1))  # copy the state N times
        trajs = [state]
        for ti in range(nt):
            state, _ = env.next_step(state, controls[:, ti])
            trajs.append(state)
        trajs = torch.stack(trajs, dim=1)  # (N, T+1, K)
        dist = torch.norm(trajs[:, -10:, 3:5]-trajs[:, -10:, -2:], dim=-1)
        loss_vec = torch.mean(dist, dim=-1) + 10*torch.mean(torch.nn.ReLU()(-controls[:, :, 0]), dim=-1)
        loss = torch.mean(loss_vec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epi, loss.item())

    min_idx = torch.argmin(loss_vec)
    print("idx=", min_idx, "loss=",loss_vec[min_idx])
    return controls[min_idx]

def main():
    random.seed(1007)
    np.random.seed(1007)
    torch.manual_seed(1007)

    env = SpaceEnv(None)

    state = env.reset()

    nt = 30

    # planning process
    info = None
    fs_list = []
    planned_charge = False
    planned_catch = False
    prev_action = None
    base_ti = 0
    for ti in range(nt * 4):
        if ti-base_ti >=nt:
            break
        if ti==0:
            controls = gradient_planner(env, nt)
            base_ti = 0
        elif info is not None and info["landed"] and not planned_charge:
            planned_charge = True
            controls = gradient_planner_v1(env, nt)
            base_ti = ti
        elif info is not None and info["charged"] and not planned_catch:
            planned_catched = True
            controls = gradient_planner_v2(env, nt)
            base_ti = ti
            
        action = controls[ti-base_ti].detach().cpu().numpy()
        if (info is None or not info["closed_enough"]) and (prev_action is None or prev_action[2] != 1.0):
            action[2] = 0.0
        else:
            action[2] = 1.0
        obs, reward, done, info = env.step(action)
        fname = env.render()
        fs_list.append(fname)
        prev_action = action
        print("ti=",ti, action[2])

    with imageio.get_writer("simulation.gif", mode='I', duration=0.1) as writer:
        for filename in fs_list:
            image = imageio.imread(filename)
            writer.append_data(image)

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Finished in %.4f seconds"%(t2-t1))