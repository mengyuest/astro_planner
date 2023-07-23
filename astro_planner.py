import time
import random
import numpy as np
import torch
import imageio
import gym
from gym import spaces
from matplotlib.patches import Polygon, Rectangle, Ellipse, Circle
import matplotlib.pyplot as plt 
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy import ndimage

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
        self.G = 1.0  # 100
        self.G1 = 1.0
        self.dt = 0.25

        # planet radius 
        self.planet_r = 3.0

        # for simulation statistics
        self.epi=0
        self.t=0
        self.time_limit=100

        self.iss_x = -8
        self.iss_y = -5

        # several orbits
        self.r_drop = self.planet_r + 1.5 
        self.r_nav = self.planet_r + 3.0

        self.picked_up = False

        self.strs = ["0-Descent Phase", "1-Orbital Maneuvering", "2-Rendezvous and Retrieval", "3-Ascent Phase"]
        self.realistic = True
        if self.realistic:
            self.iss_img = plt.imread("iss_icon.png")
            self.rocket_img = plt.imread("rocket.png")
            self.planet_img = plt.imread("neptune.png")


    def next_step(self, state, action, sim=False):
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
        
        # gravity force
        r = ((x-x0)**2+(y-y0)**2)**0.5
        rx = r * torch.cos(yaw)
        ry = r * torch.sin(yaw)
        gx = - self.G * self.planet_mass / (r)**3 * rx 
        gy = - self.G * self.planet_mass / (r)**3 * ry
        ax = acc * torch.cos(yaw) + gx
        ay = acc * torch.sin(yaw) + gy

        vx = v * torch.cos(yaw)
        vy = v * torch.sin(yaw)
        new_x = x + vx * self.dt
        new_y = y + vy * self.dt
        new_vx = vx + ax * self.dt
        new_vy = vy + ay * self.dt
        new_v = (new_vx**2+new_vy**2)**0.5
        new_yaw = yaw + w * self.dt

        # new_x = x + v * torch.cos(yaw) * self.dt
        # new_y = y + v * torch.sin(yaw) * self.dt
        # new_v = v + acc * self.dt
        # new_yaw = yaw + w * self.dt

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

        if sim:
            self.sensor_state = [mode.item(), on_surface.item()]

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
        self.energy = 0.0

        return self.state

    def step(self, action):
        self.trajs.append(self.state)
        new_state, info = self.next_step(torch.from_numpy(self.state).float().unsqueeze(0), torch.from_numpy(action).float().unsqueeze(0), sim=True)
        new_state = new_state[0]
        self.state = new_state.detach().cpu().numpy()
        self.action = action
        
        # if self.t>70:
        #     print(self.t, self.state[-4], self.state[-3])

        # design reward
        reward = -torch.norm(new_state[0:2] - new_state[3:5])
        done = self.t>=self.time_limit
        self.t+=1
        self.energy += np.linalg.norm(action)
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
        plt.figure(figsize=(8, 8))
        ax=plt.gca()

        bg = Rectangle([-15, -13], 15*2, 13*2, color="black", zorder=1)
        ax.add_patch(bg)

        # space station
        if self.realistic:
            im = OffsetImage(self.iss_img, zoom=.1)
            ab = AnnotationBbox(im, (self.iss_x, self.iss_y), frameon=False)
            ax.add_artist(ab)
        else:
            iss = Circle([self.iss_x, self.iss_y], 1, color="brown", zorder=5, alpha=0.95)
            ax.add_patch(iss)

        x0, y0, phi0, x, y, yaw, v, mode, x1, y1, vx1, vy1 = self.state

        # planet
        if self.realistic:
            im = OffsetImage(self.planet_img, zoom=.188)
            ab = AnnotationBbox(im, (x0, y0), frameon=False)
            ab.set_zorder(1)
            ax.add_artist(ab)
            
        else:
            circ=Circle([x0, y0], self.planet_r, color="royalblue", alpha=.95)
            ax.add_patch(circ)

        # phase direction
        x_from = x0
        x_to = x0 + self.planet_r * np.cos(phi0)
        y_from = y0
        y_to = y0 + self.planet_r * np.sin(phi0)
        plt.plot([x_from, x_to], [y_from, y_to], color="cyan")

        # orbits
        circ = Circle([x0, y0], self.r_drop, edgecolor="lightgray", facecolor=None, fill=False, alpha=1.0, linewidth=3.0, linestyle="-.")
        ax.add_patch(circ)

        circ = Circle([x0, y0], self.r_nav, edgecolor="darkgray", facecolor=None, fill=False, alpha=1.0, linewidth=3.0, linestyle="-.")
        ax.add_patch(circ)

        # spaceship
        if self.realistic:
            rotated_img = np.clip(np.array(ndimage.rotate(self.rocket_img, yaw*180/np.pi), np.float32), 0, 1)
            im = OffsetImage(rotated_img, zoom=.06)
            ab = AnnotationBbox(im, (x, y), frameon=False)
            ax.add_artist(ab)
        else:
            pts = self.ship_polygon(x, y, yaw)
            poly = Polygon(pts, color="lightgreen", alpha=0.95)
            ax.add_patch(poly)

        x_from = x + 0.4 * np.cos(yaw+np.pi)
        x_to = x + 0.6 * np.cos(yaw+np.pi)
        y_from = y + 0.4 * np.sin(yaw+np.pi)
        y_to = y + 0.6 * np.sin(yaw+np.pi)
        plt.plot([x_from, x_to], [y_from, y_to], color="brown")
        # trajectory
        plt.plot([xx[3] for xx in self.trajs], [xx[4] for xx in self.trajs], color="red", alpha=0.95, linestyle="-", markersize=2, marker='o')

        # sensor
        if mode==1:
            if self.picked_up or (self.ship_mode>=2 and (x-x1)**2+(y-y1)**2<0.5):
                self.picked_up=True
            else:
                pts1 = self.sensor_polygon(x1, y1)
                poly1 = Polygon(pts1, color="pink", alpha=0.9)
                ax.add_patch(poly1)

        # panel information
        planet_dx = self.state[3] - self.state[0]
        planet_dy = self.state[4] - self.state[1]
        planet_distance = (planet_dx**2+planet_dy**2)**0.5
        
        angle = self.state[5]
        speed = self.state[6]
        
        thrust = self.action[0]
        torque = self.action[1]

        if self.sensor_state[0]==0:
            state_str = "Idle"
        elif self.sensor_state[1]==0:
            state_str = "Dropped"
        else:
            if self.picked_up==False:
                state_str = "Landed"
            else:
                state_str = "Picked up"
        str_full = "Energy: %.3f\nPlanet distance: %.3f\nPlanet dx: %.3f\nPlanet dy: %.3f\nShip angle: %.3f\nShip speed: %.3f\nShip thrust: %.3f\nShip torque: %.3f\nSensor status: %s\nSensor distance: %.3f"%(
            self.energy, planet_distance, planet_dx, planet_dy, angle, speed, thrust, torque,
            state_str, ((self.state[-4]-self.state[3])**2+(self.state[4]-self.state[-3])**2)**0.5
        )
        plt.text(5, -12, str_full, bbox=dict(facecolor='gray', alpha=0.2), color="yellow", fontsize=10)
        plt.axis("scaled")
        plt.xlim(-15, 15)
        plt.ylim(-13, 13)
        plt.title("Simulation: t=%.2fs  Stage: %22s"%(self.t * self.dt, self.strs[self.ship_mode]), fontsize=16)
        filename = "sim_e%04d_t%04d.png"%(self.epi, self.t)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        return filename

def compute_angle_loss(angle, ref_angle):
    return torch.square(torch.cos(angle)-torch.cos(ref_angle)) + torch.square(torch.sin(angle)-torch.sin(ref_angle))


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
        w = 2 * np.pi / (nt * env.dt * 1.35)
        # t_tgt=30

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
            # TODO-0
            # reach the target point (tgt_x, tgt_y) at the final state (t=-1 in trajs)
            # reach the target speed tgt_v and target angle tgt_th
            # hint: the final loss should be like
            # loss = torch.mean(point_loss) + torch.mean(v_loss) + torch.mean(head_loss)

            point_loss = torch.square(trajs[-1:, 3] - tgt_x) + torch.square(trajs[-1:, 4] - tgt_y)
            v_loss = torch.square(tgt_v-controls[-1:, 0])
            angle_loss = torch.square(tgt_th-trajs[-1:, 5])
            loss = torch.mean(point_loss) + torch.mean(v_loss) + torch.mean(angle_loss)

        elif stage == 1:
            # TODO-1 
            # reach a moving target point on the orbit with radius r and with angle theta=theta0 - w * ti * env.dt for the ti timestep
            # also make sure the rocket heading is aligned with the orbit (you can use compute_angle_loss API above)
            # hint: the final loss should be like
            # loss= torch.mean(point_loss) + torch.mean(angle_loss)

            base_angle = torch.atan2(trajs[0:1, 4]-trajs[0:1, 1],trajs[0:1, 3]-trajs[0:1, 0])
            angles = base_angle + -w * torch.linspace(0, nt*env.dt, nt+1)
            target_point_x = trajs[:, 0] + r * torch.cos(angles)
            target_point_y = trajs[:, 1] + r * torch.sin(angles)
            target_point = torch.stack([target_point_x, target_point_y], dim=-1)
            target_point_head = torch.arctan2(target_point[1:, 1]-target_point[:-1, 1], target_point[1:, 0]-target_point[:-1, 0])

            point_loss = torch.norm(target_point[5:, 0:2]-trajs[5:, 3:5], dim=-1)
            angle_loss = compute_angle_loss(trajs[:-1, 5], target_point_head)
            loss= torch.mean(point_loss) + torch.mean(angle_loss)

        elif stage == 2:
            # TODO-2 
            # reach the sensor pickup point, 
            # make sure the rocket heading is aligned with the sensor moving direction (or reverse direction)
            # and not collide with the planet
            # hint: the final loss should be like
            # loss = torch.mean(point_loss) + torch.mean(angle_loss) + torch.mean(safety_loss) * weight  #, where you need to tune weight
            last_sensor_xy = trajs[-1:, -4:-2]
            last_sensor_th = torch.atan2(last_sensor_xy[:, 1]-trajs[-1:, 1], last_sensor_xy[:, 0]-trajs[-1:, 0])
            pickup_xy = torch.stack([
                trajs[-1:, 0] + (env.planet_r + 0.3) * torch.cos(last_sensor_th),
                trajs[-1:, 1] + (env.planet_r + 0.3) * torch.sin(last_sensor_th),
            ], dim=-1)

            point_loss = torch.norm(trajs[-2:, 3:5]-pickup_xy, dim=-1)

            head = last_sensor_th + np.pi/2
            angle_loss = torch.square(torch.cos(head)-torch.cos(trajs[-2:, 5])) + torch.square(torch.sin(head)-torch.sin(trajs[-2:, 5]))

            dist_planet = torch.norm(trajs[:, 3:5]-trajs[:, 0:2], dim=-1)
            safety_loss = torch.nn.ReLU()(env.planet_r - dist_planet)

            loss = torch.mean(point_loss) + torch.mean(angle_loss) + torch.mean(safety_loss) * 10
        
        elif stage == 3:
            # TODO-3 
            # reach the ISS (env.iss_x, env.iss_y)
            # stay there
            # and not colliding with the planet
            # hint: the final loss should be like
            # loss = torch.mean(point_loss) + torch.mean(safety_loss) * weight + torch.mean(smooth_loss) #, where you need to tune weight
            tgt_xy = torch.stack([
                trajs[-1:, 0] * 0 + env.iss_x,
                trajs[-1:, 1] * 0 + env.iss_y,
            ], dim=-1)

            point_loss = torch.norm(trajs[-5:, 3:5]-tgt_xy, dim=-1) + trajs[-5:, 6]**2
            dist_planet = torch.norm(trajs[:, 3:5]-trajs[:, 0:2], dim=-1)
            safety_loss = torch.nn.ReLU()(env.planet_r - dist_planet)
            smooth_loss = torch.norm(controls[:, 0:2], dim=-1)
            loss = torch.mean(point_loss) + torch.mean(safety_loss) * 10 + torch.mean(smooth_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epi, loss.item())
    
    if stage==0:
        controls.detach()[-1, 2] = 1.0

    return controls

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
    for ti in range(200):
        env.ship_mode = ship_mode
        if ship_mode == 4:
            break
        # planning for control
        if not is_planned:
            is_planned = True
            if ship_mode == 0:
                horizon_t = 30
                controls = gradient_planner(env, horizon_t, stage=0, num_iters=100, lr=0.1)
            elif ship_mode == 1:
                horizon_t = 50
                controls = gradient_planner(env, horizon_t, stage=1, num_iters=100, lr=0.05)
            elif ship_mode == 2:
                horizon_t = 30
                controls = gradient_planner(env, horizon_t, stage=2, num_iters=100, lr=0.1)
            else:
                horizon_t = 40
                controls = gradient_planner(env, horizon_t, stage=3, num_iters=300, lr=0.05)
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
        
        print("ti=",ti)

    with imageio.get_writer("simulation.gif", mode='I', duration=0.1) as writer:
        for filename in fs_list:
            image = imageio.imread(filename)
            writer.append_data(image)

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Finished in %.4f seconds"%(t2-t1))