import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# create a new directory for the run under 'datasets/simulation_data'
import datetime
path_to_runs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'datasets/simulation_data')
tool_type = 'medium'
NUM_EP = 2000
collecting = True
range_start = 100
range_end = 120
limited = True
if limited:
    run_name = f'{tool_type}_tool_limited_{range_start}_{range_end}disks_{NUM_EP}ep'
else:
    run_name = f'{tool_type}_tool_{range_start}_{range_end}disks_{NUM_EP}ep'
run_dir = os.path.join(path_to_runs, run_name)
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

from ToolUser.config import STATE_DIM, HIDDEN_SIZE, MAX_STEP, np
import ToolUser.config as config

from ToolUser.utils import state_transformer, state_transformer_sweep, square_state_transformer
from ToolUser.train import DatasetCollector 
import torch
from tqdm import trange
from ToolUser.buffer import ReplayBuffer
# watch a specific transition from a saved buffer
if False:
    render = False
    file_name = 'buffer_39.pt'
    buffer = torch.load(os.path.join(run_dir, file_name), weights_only=False)
    transition = buffer[4501]
    state = transition[0].squeeze(0).numpy()
    action = transition[1].squeeze(0).numpy()
    print("State:",state[0:3])
    print("Action:",action)
    print("Next State:",transition[3].squeeze(0).numpy()[0:3])
    config.DISK_NUM = 39
    collector = DatasetCollector(render=render)
    collector.env.reset(state = state)
    nxt = collector.env.step_no_goal(state, action,render=render)
    tool_pts = np.loadtxt(f'object_outline_normalized_{tool_type}.txt' , dtype=np.float32)
    transition= (state,action,0,torch.tensor(nxt,dtype=torch.float32),0,0)
    _ = state_transformer([transition],sigma=config.R/5,disk_count=config.DISK_NUM, tool_pts=tool_pts)
    sys.exit()




with trange(range_start,range_end, desc='Transforming Data') as t:
# with trange(30,31) as t:
    for i in t:
        buffer = None
        config.DISK_NUM = i
        config.TOOL_PTS = f'object_outline_{tool_type}.txt'
        # print(f"Collecting data for {config.DISK_NUM} disks...")
        # buffer = trainer.collect_demonstrations(episodes=100, max_step=1, render=False)
        # buffer = trainer.collect_explorations(episodes=100, max_step=MAX_STEP, render=False)#,buffer = buffer)
        file_name = f'buffer_{config.DISK_NUM}.pt'
        if collecting:
            collector = DatasetCollector(render=False)
            if limited:
                buffer = collector.collect_interactions_trans_limited(episodes=NUM_EP, buffer=buffer,limit=0.05)
            else:
                buffer = collector.collect_interactions_trans(episodes=NUM_EP, buffer=buffer)
            buffer.save(os.path.join(run_dir, file_name))
        else:
            buffer = torch.load(os.path.join(run_dir, file_name), weights_only=False)
        
        tool_pts = np.loadtxt(f'object_outline_normalized_{tool_type}.txt' , dtype=np.float32)
        # buffer = square_state_transformer(old_boundaries=(480,320), buffer=buffer, batch_size=64)
        # buffer.save(os.path.join(run_dir, file_name.replace('.pt','_square.pt')))
        buffer = state_transformer_sweep(buffer,sigma=config.R/5,disk_count=config.DISK_NUM, tool_pts=tool_pts)
        field_file_name = f'buffer_sweepfield_{config.DISK_NUM}.pt'
        buffer.save(os.path.join(run_dir,field_file_name))

        


# trainer.pretrain_actor_IL(dem_buffer=buffer,epochs=200)
# trainer.pretrain_predictor(buffer=buffer, epochs=2000)
# trainer.train_RL()

