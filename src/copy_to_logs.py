import torch
import os

path = "/home/llnu/private/Itachi/DL3D/models/bottles_30000_1.0_100_5_1.tar"
accum_iter = 2
di = torch.load(path, map_location = torch.device('cpu'))
with open("logs/logs.txt", 'w') as f:
    for iteration in range(len(di['loss_history_color'])):
        if iteration % (10*accum_iter) == 0:
            f.write(
                f'Iteration {iteration:05d}:'
                + f" loss color = {float(di['loss_history_color'][iteration]):1.2e}"
                + f" psnr = {float(di['loss_history_psnr'][iteration]):1.2e}"
                + '\n'
            )