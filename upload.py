# Uploads targetted runs to server
# Requires SSH keys to be configure on host and remote!
# Use: ssh-copy-id sammy@your_server_address

import os
import subprocess
from dirsync import sync
import constants as c


# Define experiment regions
regions = [c.florida, c.central, c.west]

# Define experiments shortnames
exps = ["CNN", "13x13", "9x9", "5x5", "nf"]

# def sync(source, target):
#     subprocess.call([
#         "rsync",
#         "-arvz",
#         "--progress",
#         source,
#         f"{os.getenv('USER')}@{os.getenv('REMOTE')}:{target}"
#     ])

# with pysftp.Connection(os.getenv("REMOTE"), username=os.getenv("USER")) as sftp:
for r in regions:
    for e in exps:
        exp_num = next(iter(r[e]))
        source_path = f"{os.getenv('EXPERIMENT_DATA_DIR')}/{exp_num}/{r[e][exp_num]}"
        target_path = f"{os.getenv('SLURM_TMPDIR')}/{exp_num}/{r[e][exp_num]}/"
        print("SOURCE: ", source_path)
        print("TARGET: ", target_path)
        sync(source_path, target_path, 'sync')
