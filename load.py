

from envs.params import save_path

from envs.tools import load_bbx, load_pcd, Vis


import os
import re

vis = Vis()

for episode_i in os.listdir(save_path):
    files = os.listdir(os.path.join(save_path, episode_i))
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    bbx_files = [f for f in files if f.startswith('bbx')]
    pcd_files = [f for f in files if f.startswith('pcd')]

    for bbx_file, pcd_file in zip(bbx_files, pcd_files):
        bbx = load_bbx(os.path.join(save_path, episode_i, bbx_file))
        pcd = load_pcd(os.path.join(save_path, episode_i, pcd_file))

        vis.run_step(bbx, pcd)


vis.destroy()


