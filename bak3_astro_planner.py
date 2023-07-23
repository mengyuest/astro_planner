
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
        ship_dim = 4  # (x, y, phi, v)
        target_dim = 5
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
        # self.K = 0.05
        self.K = 0.00
        self.K1 = 0.3

        #self.G = 6.67430e-11
        self.G = 0.0  # 100
        self.G1 = 1.0
        self.dt = 0.25

        # planet radius 
        self.planet_r = 3.0

        # for simulation statistics
        self.epi=0
        self.t=0
        self.time_limit=100

        # several orbits
        self.r_drop = self.planet_r + 1.5 
        self.r_nav = self.planet_r + 3.0
    
    def next_step(self, state, action):
        # state representation2
        # (x0, y0, phi0) position and rotation phase of the planet
        # (x, y, yaw) position and heading of the spaceship
        # (mode, x1, y1, vx1, vy1) the mode and coordinate for the sensor
        x0, y0, phi0, x, y, yaw, v, mode, x1, y1, vx1, vy1 = torch.split(state, split_size_or_sections=1, dim=-1)

        info = {}
        
        dt = self.dt

        th = torch.atan2(y0 / self.b, x0 / self.a)
        th = (th + self.K * dt)
        th = torch.atan2(torch.sin(th), torch.cos(th))
        
        new_x0 = self.a * torch.cos(th) 
        new_y0 = self.b * torch.sin(th)
        new_phi0 = phi0 + dt * self.K1

        acc = action[:, 0:1]
        w = action[:, 1:2]
        mode = torch.clip(mode + action[:, 2:3], 0, 1)
        
        new_x = x + v * torch.cos(yaw) * self.dt
        new_y = y + v * torch.sin(yaw) * self.dt
        new_v = v + acc * self.dt
        new_yaw = yaw + w * self.dt

        # consider free-fall motion
        theta_01 = torch.atan2(y1-y0, x1-x0)
        r_01 = ((x1-x0)**2+(y1-y0)**2)**0.5
        rx = r_01 * torch.cos(theta_01)
        ry = r_01 * torch.sin(theta_01)
        ax1 = - self.G1 * self.planet_mass / (r_01)**3 * rx 
        ay1 = - self.G1 * self.planet_mass / (r_01)**3 * ry

        new_x1_ff = x1 + vx1 * dt
        new_y1_ff = y1 + vy1 * dt
        new_vx1_ff = vx1 + ax1 * dt
        new_vy1_ff = vy1 + ay1 * dt
        # # rotation transform
        # new_x1_ff1 = (new_x1_ff - new_x0) * np.cos(dt * self.K1) - (new_y1_ff - new_y0) * np.sin(dt * self.K1) + new_x0
        # new_y1_ff1 = (new_x1_ff - new_x0) * np.sin(dt * self.K1) + (new_y1_ff - new_y0) * np.cos(dt * self.K1) + new_y0
        # new_vx1_ff1 = new_vx1_ff * np.cos(dt * self.K1) - new_vy1_ff * np.sin(dt * self.K1)
        # new_vy1_ff1 = new_vx1_ff * np.sin(dt * self.K1) + new_vy1_ff * np.cos(dt * self.K1)
        # new_x1_ff, new_y1_ff, new_vx1_ff, new_vy1_ff = new_x1_ff1, new_y1_ff1, new_vx1_ff1, new_vy1_ff1

        # on ground clip
        on_surface = (((x1-x0)**2 + (y1-y0)**2)**0.5 <= self.planet_r+0.01).float()
        new_theta_01 = theta_01 + dt * self.K1
        
        new_x1_gnd = new_x0 + self.planet_r * torch.cos(new_theta_01)
        new_y1_gnd = new_y0 + self.planet_r * torch.sin(new_theta_01)
        new_vx1_gnd = 0 * vx1
        new_vy1_gnd = 0 * vy1
        
        new_x1 = (mode==0).float() * new_x + (mode!=0).float() * (on_surface * new_x1_gnd + (1-on_surface) * new_x1_ff)
        new_y1 = (mode==0).float() * new_y + (mode!=0).float() * (on_surface * new_y1_gnd + (1-on_surface) * new_y1_ff)
        new_vx1 = (mode==0).float() * vx1 + (mode!=0).float() * (on_surface * new_vx1_gnd + (1-on_surface) * new_vx1_ff)
        new_vy1 = (mode==0).float() * vy1 + (mode!=0).float() * (on_surface * new_vy1_gnd + (1-on_surface) * new_vy1_ff)

        new_state = torch.cat([
            new_x0, new_y0, new_phi0, 
            new_x, new_y, new_yaw, new_v,
            mode, new_x1, new_y1, new_vx1, new_vy1
        ], dim=-1)

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
        vx1, vy1 = 0, 0
        
        self.epi += 1
        self.t = 0
        self.trajs = []
        self.state = np.array([
            x0, y0, phi0, x, y, yaw, v, mode, x1, y1, vx1, vy1
        ]).astype(np.float32)

        return self.state

    def step(self, action):
        self.trajs.append(self.state)
        new_state, info = self.next_step(torch.from_numpy(self.state).float().unsqueeze(0), torch.from_numpy(action).float().unsqueeze(0))
        new_state = new_state[0]
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
        points = (scale * points.T).T + np.array([[x, y]])
        return points

    def render(self):
        plt.figure(figsize=(10, 10))
        # orbit
        ns = 1000
        ths = np.linspace(0, np.pi*2, ns)
        xs = self.a * np.cos(ths)
        ys = self.b * np.sin(ths)
        plt.plot(xs, ys, color='gray', alpha=0.5, linewidth=3.0)

        x0, y0, phi0, x, y, yaw, v, mode, x1, y1, vx1, vy1 = self.state

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

        # orbits
        circ = Circle([x0, y0], self.r_drop, edgecolor="black", facecolor=None, alpha=0.4, linewidth=2.0, linestyle="-.")
        ax.add_patch(circ)

        circ = Circle([x0, y0], self.r_nav, edgecolor="purple", facecolor=None, alpha=0.4, linewidth=2.0, linestyle="-.")
        ax.add_patch(circ)

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
        plt.title("Simulation %03d  mode %d"%(self.t, self.state[7]))
        filename = "sim_e%04d_t%04d.png"%(self.epi, self.t)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        return filename


def gradient_planner(env, nt, stage, num_iters, lr):
    if stage == 0:
        x0, y0 = env.state[0], env.state[1]
        th = np.arctan2(y0 / env.b, x0 / env.a)
        th = (th + env.K * env.dt * nt)
        th = np.arctan2(np.sin(th), np.cos(th))    
        new_x0 = env.a * np.cos(th) 
        new_y0 = env.b * np.sin(th)
        tgt_x, tgt_y, tgt_v, tgt_th = get_ref(env.state[3], env.state[4], new_x0, new_y0, env.r_drop)
        # print(tgt_x, tgt_y)
        # exit()

    elif stage == 1:
        r = env.r_nav
        w = 2 * np.pi / (nt * env.dt * 2)

    controls = torch.zeros(nt, 3, requires_grad=True)
    optimizer = torch.optim.Adam([controls], lr=lr)

    for epi in range(num_iters):
        state = torch.from_numpy(env.state).float().to(controls.device).unsqueeze(0)
        trajs=[state]
        for ti in range(nt):
            state, _ = env.next_step(state, controls[ti:ti+1])
            trajs.append(state)
        trajs = torch.stack(trajs, dim=1)[0]

        # compute the loss
        if stage == 0:
            point_loss = torch.square(trajs[-1:, 3] - tgt_x) + torch.square(trajs[-1:, 4] - tgt_y)
            v_loss = torch.square(tgt_v-controls[-1:, 0])
            head_loss = torch.square(tgt_th-trajs[-1:, 5])
            loss = torch.mean(point_loss) + torch.mean(v_loss) + torch.mean(head_loss)

        elif stage == 1:
            base_phase = torch.atan2(trajs[0:1, 4],trajs[0:1, 3])
            phases = base_phase + -w*torch.linspace(0, nt*env.dt, nt+1)
            target_point_x = trajs[:, 0] + r * torch.cos(phases)
            target_point_y = trajs[:, 1] + r * torch.sin(phases)
            target_point = torch.stack([target_point_x, target_point_y], dim=-1)
            target_point_head = torch.arctan2(target_point[1:, 1]-target_point[:-1, 1], target_point[1:, 0]-target_point[:-1, 0])

            point_loss = torch.norm(target_point-trajs[:, 3:5], dim=-1)
            head_loss = torch.square(torch.cos(trajs[:-1, 5])-torch.cos(target_point_head)) + torch.square(torch.sin(trajs[:-1, 5])-torch.sin(target_point_head))
            loss= torch.mean(point_loss) + torch.mean(head_loss)

        elif stage == 2:
            last_sensor_xy = trajs[-1:, -4:-2]
            last_sensor_th = torch.atan2(last_sensor_xy[:, 1], last_sensor_xy[:, 0])
            pickup_xy = torch.stack([
                trajs[-1:, 0] + (env.planet_r + 0.3) * torch.cos(last_sensor_th),
                trajs[-1:, 1] + (env.planet_r + 0.3) * torch.sin(last_sensor_th),
            ], dim=-1)

            loss_point = torch.norm(trajs[-2:, 3:5]-pickup_xy, dim=-1)

            head = last_sensor_th + np.pi/2
            loss_head = torch.square(torch.cos(head)-torch.cos(trajs[-2:, 5])) + torch.square(torch.sin(head)-torch.sin(trajs[-2:, 5]))

            dist_planet = torch.norm(trajs[:, 3:5]-trajs[:, 0:2], dim=-1)
            loss_safe = torch.nn.ReLU()(env.planet_r - dist_planet)

            loss = torch.mean(loss_point) + torch.mean(loss_head) + torch.mean(loss_safe)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epi, loss.item())
    
    if stage==0:
        controls.detach()[-1, 2] = 1.0
    return controls

# def gradient_planner_v0(env, nt):
#     # N = 20
#     N = 1
#     target_phi = torch.linspace(0, np.pi * 2, N)
#     controls = torch.zeros((N, nt, 3))
#     controls.requires_grad = True
#     optimizer = torch.optim.Adam([controls], lr=lr)
#     for epi in range(num_iters):
#         state = torch.from_numpy(env.state).float()
#         state = torch.tile(state.unsqueeze(0), (N, 1))  # copy the state N times
#         trajs = [state]
#         for ti in range(nt):
#             state, _ = env.next_step(state, controls[:, ti])
#             trajs.append(state)
#         trajs = torch.stack(trajs, dim=1)  # (N, T+1, K)

#         target_phase = target_phi.reshape((N, 1))
#         target_point_x = trajs[:, :, 0]*0 + tgt_x
#         target_point_y = trajs[:, :, 1]*0 + tgt_y
#         target_point = torch.stack([target_point_x, target_point_y], dim=-1)

#         point_loss = torch.mean(torch.norm(target_point[:, -2:, 0:2]-trajs[:, -2:, 3:5], dim=-1), dim=-1)
#         v_loss = torch.mean(torch.square(tgt_v-controls[:,-2:,0]), dim=-1)
#         head_loss = torch.mean(torch.square(tgt_th-trajs[:, -2:, 5]), dim=-1)
#         norm_loss = torch.mean(torch.norm(controls, dim=-1), dim=-1)
#         loss_vec = point_loss + v_loss + head_loss #+ norm_loss * 0.01
#         loss = torch.mean(loss_vec)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print(epi, loss.item())

#     min_idx = torch.argmin(loss_vec)
#     controls.detach()[:, -1, 2] = 1.0
#     return controls[min_idx]

# def gradient_planner_v1(env, nt, ):
#     # the goal is to reach a higher orbit, at what angular speed, and maintain for how long
#     num_iters = 200
#     lr = 0.1

#     r = env.r_nav
#     T = nt * env.dt * 2
#     w = 2 * np.pi / T

#     # N = 20
#     N = 1
#     target_phi = torch.linspace(0, np.pi * 2, N)
#     controls = torch.zeros((N, nt, 3))
#     controls.requires_grad = True
#     optimizer = torch.optim.Adam([controls], lr=lr)
#     for epi in range(num_iters):
#         state = torch.from_numpy(env.state).float()
#         state = torch.tile(state.unsqueeze(0), (N, 1))  # copy the state N times
#         trajs = [state]
#         for ti in range(nt):
#             state, _ = env.next_step(state, controls[:, ti])
#             trajs.append(state)
#         trajs = torch.stack(trajs, dim=1)  # (N, T+1, K)

#         # phase_offset = np.pi / 6
#         # target_phase = target_phi.reshape((N, 1)) + phase_offset
#         # target_point_x = trajs[:, :, 0] + env.r_nav * torch.cos(target_phase+trajs[:, :, 2])
#         # target_point_y = trajs[:, :, 1] + env.r_nav * torch.sin(target_phase+trajs[:, :, 2])
#         base_phase = torch.atan2(trajs[:, 0:1, 4],trajs[:, 0:1, 3])
#         phases = torch.linspace(0, nt*env.dt, nt+1)
#         phases = base_phase + phases * (-w)
#         target_point_x = trajs[:, :, 0] + r * torch.cos(phases)
#         target_point_y = trajs[:, :, 1] + r * torch.sin(phases)
#         target_point = torch.stack([target_point_x, target_point_y], dim=-1)
#         target_point_head = torch.arctan2(target_point[:, 1:, 1]-target_point[:, :-1, 1], target_point[:, 1:, 0]-target_point[:, :-1, 0])

#         # radius = torch.norm(trajs[:, :, 3:5]-trajs[:,:,0:2], dim=-1)
#         # point_loss = torch.mean(torch.square(radius-env.r_nav), dim=-1)
#         point_loss = torch.mean(torch.norm(target_point-trajs[:,:,3:5], dim=-1), dim=-1)
#         # neg_speed_loss = torch.mean(torch.nn.ReLU()(-trajs[:, :, 6]), dim=-1)
#         # large_angle_loss = torch.mean(torch.square(controls[:, :, 1]), dim=-1)
#         # loss_vec = point_loss + neg_speed_loss + large_angle_loss

#         # head_loss = torch.mean(torch.square(trajs[:, :-1, 5]-target_point_head), dim=-1)

#         head_loss = torch.mean(torch.square(torch.cos(trajs[:, :-1, 5])-torch.cos(target_point_head)), dim=-1) + torch.mean(torch.square(torch.sin(trajs[:, :-1, 5])-torch.sin(target_point_head)), dim=-1)

#         loss_vec = point_loss + head_loss
#         loss = torch.mean(loss_vec)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print(epi, loss.item())

#     min_idx = torch.argmin(loss_vec)
#     print("idx=", min_idx, "loss=",loss_vec[min_idx])
#     return controls[min_idx]

# def gradient_planner_v2(env, nt, ):
#     num_iters = 100
#     lr = 0.1

#     # N = 20
#     # target_phi = torch.linspace(0, np.pi * 2, N)
#     # controls = torch.zeros((N, nt, 3))
#     N = 1
#     controls = torch.zeros((N, nt, 3))
#     controls.requires_grad = True
#     optimizer = torch.optim.Adam([controls], lr=lr)
#     for epi in range(num_iters):
#         state = torch.from_numpy(env.state).float()
#         state = torch.tile(state.unsqueeze(0), (N, 1))  # copy the state N times
#         trajs = [state]
#         for ti in range(nt):
#             state, _ = env.next_step(state, controls[:, ti])
#             trajs.append(state)
#         trajs = torch.stack(trajs, dim=1)  # (N, T+1, K)

#         last_sensor_xy = trajs[:, -1:, -4:-2]
#         last_sensor_th = torch.atan2(last_sensor_xy[:, :, 1], last_sensor_xy[:, :, 0])
#         pickup_xy = torch.stack([
#             trajs[:, -1:, 0] + (env.planet_r + 0.3) * torch.cos(last_sensor_th),
#             trajs[:, -1:, 1] + (env.planet_r + 0.3) * torch.sin(last_sensor_th),
#         ], dim=-1)

#         dist = torch.norm(trajs[:, -2:, 3:5]-pickup_xy, dim=-1)
#         head = last_sensor_th + np.pi/2

#         loss_point = torch.mean(dist, dim=-1)
#         loss_head = torch.mean(torch.square(torch.cos(head)-torch.cos(trajs[:, -2:, 5])), dim=-1) + torch.mean(torch.square(torch.sin(head)-torch.sin(trajs[:, -2:, 5])), dim=-1)

#         dist_planet = torch.norm(trajs[:, :, 3:5]-trajs[:, :, 0:2], dim=-1)
#         loss_safe = torch.mean(torch.nn.ReLU()(env.planet_r - dist_planet), dim=-1)

#         loss_vec = loss_point + loss_head + loss_safe
#         loss = torch.mean(loss_vec)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print(epi, loss.item())

#     min_idx = torch.argmin(loss_vec)
#     print("idx=", min_idx, "loss=",loss_vec[min_idx])
#     return controls[min_idx]

# https://stackoverflow.com/questions/1351746/find-a-tangent-point-on-circle
def get_ref(px, py, cx, cy, r):
    points = np.zeros(4)
    dx = cx - px
    dy = cy - py
    pc2 = dx**2 + dy**2
    pc = pc2**0.5
    r2 = pc2 - r**2
    d = r2 / pc
    h = (r2 - d**2)**0.5
    points[0] = px + (dx * d - dy * h) / pc
    points[1] = py + (dy * d + dx * h) / pc
    points[2] = px + (dx * d + dy * h) / pc
    points[3] = py + (dy * d - dx * h) / pc

    ref_th = np.arctan2(points[1]-py, points[0]-px)
    ref_v = 3.0

    return points[0], points[1], ref_v, ref_th


def main():
    random.seed(1007)
    np.random.seed(1007)
    torch.manual_seed(1007)

    env = SpaceEnv(None)
    state = env.reset()

    fs_list = []    
    base_ti = 0
    ship_mode = 0
    is_planned = False
    for ti in range(150):
        if ship_mode == 3:
            break
        # planning for control
        if not is_planned:
            is_planned = True
            if ship_mode == 0:
                horizon_t = 30
                controls = gradient_planner(env, horizon_t, stage=0, num_iters=100, lr=0.05)
            elif ship_mode == 1:
                horizon_t = 50
                controls = gradient_planner(env, horizon_t, stage=1, num_iters=200, lr=0.1)
            else:
                horizon_t = 30
                controls = gradient_planner(env, horizon_t, stage=2, num_iters=100, lr=0.1)
            base_ti = ti
        
        # interact with env
        action = controls[ti-base_ti].detach().cpu().numpy()        
        obs, reward, done, info = env.step(action)
        fname = env.render()
        fs_list.append(fname)
        
        # change ship mode
        if ti + 1 >= base_ti + horizon_t:
            ship_mode += 1
            is_planned = False
        
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
    