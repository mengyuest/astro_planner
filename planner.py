from astro2d_env import SpacecraftEnv2D
import numpy as np
import torch
import imageio

env = SpacecraftEnv2D(None)

env.reset()

nt = 100
num_iters = 100
lr = 0.1

N = 20
target_phi = torch.linspace(0, np.pi * 2, N)
controls = torch.zeros((N, nt, 3))
controls.requires_grad = True

optimizer = torch.optim.Adam([controls], lr=lr)

for epi in range(num_iters):
    state = torch.from_numpy(env.observation).float()
    state = torch.tile(state.unsqueeze(0), (N, 1))
    trajs=[state]
    for ti in range(nt):
        state = env.next_step(state, controls[:, ti], env.dt)
        trajs.append(state)
    trajs = torch.stack(trajs, dim=1)  # (N, T+1, K)

    target_radius = env.planet_r + 0.5
    target_phase = target_phi.reshape((N, 1))
    target_point_x = trajs[:, :, 0] + target_radius * torch.cos(target_phase+trajs[:, :, 2])
    target_point_y = trajs[:, :, 1] + target_radius * torch.sin(target_phase+trajs[:, :, 2])
    target_point = torch.stack([target_point_x, target_point_y], dim=-1)
    
    # ego_angle = trajs[:, 5]
    # target_angle = trajs[:, 2] + np.pi/2
    # is_close = (torch.norm(target_point - trajs[:, 3:5], dim=-1) < target_radius + 0.3).float()

    # loss = torch.norm(trajs[:, 0:3] - trajs[:, 4:7], dim=-1)
    dist = torch.norm(trajs[:, -5:, 0:2] - trajs[:, -5:, 3:5], dim=-1)
    safe_loss = torch.nn.ReLU()(env.planet_r + 0.01 - dist) * 1000
    norm_loss = torch.mean(torch.norm(controls, dim=-1), dim=-1) * 0.1
    loss_vec = torch.mean(torch.norm(target_point - trajs[:, :, 3:5], dim=-1),dim=-1) + torch.mean(safe_loss, dim=-1) + norm_loss #+ torch.square(ego_angle-target_angle) * is_close
    loss = torch.mean(loss_vec) #+ torch.norm(controls) * 0.01

    # loss = torch.mean(torch.norm(trajs[:, :, 3:5] - 0, dim=-1)) + torch.mean(torch.norm(trajs[:, 8]))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(epi, loss.item())

min_idx = torch.argmin(loss_vec)
print(min_idx)
print("Phase=",target_phi[min_idx])
# print(trajs[:, 7:10])
fs_list = []
for ti in range(nt):
    action = controls[min_idx, ti].detach().cpu().numpy() #(np.random.rand(7, ) - 0.5)*0
    obs, reward, done, info = env.step(action)
    fname = env.render()
    fs_list.append(fname)

with imageio.get_writer("simulation.gif", mode='I', duration=0.1) as writer:
    for filename in fs_list:
        image = imageio.imread(filename)
        writer.append_data(image)



