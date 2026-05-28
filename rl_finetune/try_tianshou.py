import gym
import Callienv.envs.tools as tools

from gym.wrappers.record_video import RecordVideo  ##

import torch
import time
import numpy as np
import random
import os
import json
import utils
from functools import partial

from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any
from tianshou.utils import TensorboardLogger
from tianshou.data import Batch, Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv #SubprocEnv
from tianshou.policy import BasePolicy, SACPolicy
import tianshou as ts

from tianshou.trainer.offpolicy import OffpolicyTrainer  # class
from tianshou.trainer import offpolicy_trainer  #Wrapper for OffPolicyTrainer run method.

from MLP.model import My_MLP, My_Siren
from tianshou.utils.net.common import Net, DataParallelNet
from tianshou.utils.net.continuous import ActorProb, Critic  ## study!!!

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ZeroActionPolicy(BasePolicy):
    """Policy used only to populate the initial replay buffer on the skeleton."""

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim

    def forward(self, batch: Batch, state: Any = None, **kwargs: Any) -> Batch:
        action = np.zeros((len(batch.obs), self.action_dim), dtype=np.float32)
        return Batch(act=action, state=state)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        return {}


def initialize_actor_near_zero(actor: ActorProb, initial_std: float) -> None:
    """Start SAC exploration around the skeleton-aligned zero action."""
    if initial_std <= 0:
        raise ValueError(f"--initial_actor_std must be > 0, got {initial_std}.")

    output_layers = [module for module in actor.mu.modules() if isinstance(module, torch.nn.Linear)]
    if not output_layers:
        raise ValueError("Unable to locate the ActorProb mean output layer.")

    output_layer = output_layers[-1]
    torch.nn.init.zeros_(output_layer.weight)
    if output_layer.bias is not None:
        torch.nn.init.zeros_(output_layer.bias)
    with torch.no_grad():
        actor.sigma_param.fill_(float(np.log(initial_std)))
    print(f"Initialized actor action mean to zero with std={initial_std}.")


def should_record_video_episode(episode_id: int, early_episodes: int, interval: int) -> bool:
    """Record dense early episodes and periodic later snapshots."""
    return episode_id < early_episodes or episode_id % interval == 0


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(("Continuous SAC using Tianshou, "
        "for the meaning of some hyper-parameters, "
        "refer to the documentation of Tianshou."
    ))
    # 区分epoch, episode, step
    # Network parameters
    parser.add_argument('--actor_network_shape', default=(128, 256), type=list)
    parser.add_argument('--critic_network_shape', default=(128, 256, 256), type=list)
    parser.add_argument('--learn_fourier', default = False, type=bool)
    parser.add_argument('--learn_rbf', action='store_true')
    parser.add_argument('--sigma', default = 0.1, type=float)
    parser.add_argument('--train_B', default=True, type=bool)
    parser.add_argument('--fourier_dim', default=256, type=int)
    parser.add_argument('--concatenate_fourier', action='store_true')
    parser.add_argument('--no-concatenate_fourier', dest='concatenate_fourier', action='store_false')
    parser.set_defaults(concatenate_fourier=True)
    parser.add_argument('--rbf_dim', default=256, type=int)
    parser.add_argument('--rbf_sigma', default=0.5, type=float)
    parser.add_argument('--train_rbf_centers', dest='train_rbf_centers', action='store_true')
    parser.add_argument('--freeze_rbf_centers', dest='train_rbf_centers', action='store_false')
    parser.add_argument('--concatenate_rbf', action='store_true')
    parser.add_argument('--no-concatenate_rbf', dest='concatenate_rbf', action='store_false')
    parser.set_defaults(train_rbf_centers=True, concatenate_rbf=True)

    # SAC policy
    parser.add_argument('--mu_std_net', default=(64,), type=list)
    parser.add_argument('--actor_lr', default=3e-5, type=float)
    parser.add_argument('--critic_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--tau',  default=0.005, type=float)
    parser.add_argument('--gamma', default=0.9, type=float)
    parser.add_argument('--target_entropy_ratio', default=0.98, type=float)
    parser.add_argument('--buffer_size', default=2**20, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument(
        '--initial_collect_mode',
        choices=('zero', 'random', 'policy'),
        default='zero',
        help="How to fill the initial replay buffer before SAC updates start.",
    )
    parser.add_argument('--zero_init_actor', dest='zero_init_actor', action='store_true')
    parser.add_argument('--no-zero_init_actor', dest='zero_init_actor', action='store_false')
    parser.add_argument('--initial_actor_std', default=0.05, type=float)
    parser.set_defaults(zero_init_actor=True)

    # simulator env nums
    # notice The number of train/test envs must be a divisor of the number of training/test data!
    parser.add_argument('--train_env_num', default=4, type=int)
    parser.add_argument('--test_env_num', default=4, type=int)

    # render mode: "rgb_array": record vids; "human": online renderer
    parser.add_argument('--test_render_mode', default='rgb_array', type=str)
    parser.add_argument('--train_render_mode', default='rgb_array', type=str)

    # data dirs
    ## both img and skel are in the same folder
    parser.add_argument('--train_data_dir',  default=None, type=str)
    parser.add_argument('--test_data_dir',  default=None, type=str)
    parser.add_argument(
        '--skel_datatype',
        default=None,
        type=str,
        help="Skeleton file extension in data dirs: 'npy' or 'npz'. If omitted, auto-detect in CalliEnv.",
    )

    # save dirs
    parser.add_argument('--save_video_dir', default='./result/demo/', type=str)
    parser.add_argument('--save_model_dir', default='./result/models/demo/', type=str)
    parser.add_argument('--save_control_dir', default='./result/demo/arrays/', type=str)
    parser.add_argument('--test_save_dir', default='./result/demo/test/', type=str)
    parser.add_argument('--save_visualize_dir', default='./result/demo/vis/', type=str)
    parser.add_argument('--logdir',  default=None, type=str)
    parser.add_argument(
        '--early_video_episodes',
        default=10,
        type=int,
        help="Record every train/test video episode below this episode index.",
    )
    parser.add_argument(
        '--video_episode_interval',
        default=5,
        type=int,
        help="Record later video episodes at this interval.",
    )
    parser.add_argument(
        '--checkpoint_interval',
        default=1,
        type=int,
        help="Save an epoch-numbered policy checkpoint at this epoch interval.",
    )
    parser.add_argument(
        '--early_artifact_episodes',
        default=10,
        type=int,
        help="Save control arrays and visualizations for every completed early episode.",
    )
    parser.add_argument(
        '--artifact_episode_interval',
        default=5,
        type=int,
        help="After early artifact snapshots, save one at this episode interval.",
    )

    # tool properties
    parser.add_argument('--which_tool', default='brush', type=str)
    parser.add_argument('--tool_property_dir',  default=None, type=str) ##file recording tool property

    # training
    ## training iterations
    parser.add_argument('--image_iter', default=10, type=int)
    parser.add_argument('--start_update', default=1, type=int)
    parser.add_argument('--update', default=1, type=int)
    parser.add_argument('--smooth_action_weight', default=0.01, type=float)
    parser.add_argument('--smooth_theta_weight', default=0.02, type=float)
    parser.add_argument('--smooth_pos_weight', default=0.02, type=float)
    parser.add_argument('--smooth_penalty_max', default=0.1, type=float)
    

    parser.add_argument('--max_epoch',  default=150, type=int)
    parser.add_argument('--step_per_epoch',  default=10000, type=int) #一个epoch 最多collect多少个transitions
    parser.add_argument('--batch_size',  default=2**11, type=int)
    parser.add_argument('--resume_from_log',  default=None)

    ## step_per_collect & update_per_step: the former hyper-param is used in sampling while the latter is for updating the model.
    parser.add_argument('--step_per_collect',  default=4, type=int)   #相当于走4个step，更新一次网络参数
    parser.add_argument('--update_per_step',  default=2, type=float)

    # testing
    parser.add_argument('--episode_per_test',  default=10, type=int)
    
    args, unknown = parser.parse_known_args()
    
    return args


def validate_positive_int(name, value):
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")


def validate_nonnegative_int(name, value):
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")


def validate_data_dir(name, data_dir):
    if data_dir is None:
        raise ValueError(f"--{name}_data_dir is required.")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"--{name}_data_dir does not exist or is not a directory: {data_dir!r}")


def validate_data_split(name, data_dir, image_num, env_num):
    if image_num <= 0:
        raise ValueError(
            f"--{name}_data_dir={data_dir!r} contains no .png files. "
            "Expected matching numbered .png skeleton pairs."
        )
    if image_num % env_num != 0:
        raise ValueError(
            f"--{name}_data_dir={data_dir!r} has {image_num} images, but --{name}_env_num={env_num}. "
            f"The env count must divide the image count; use --{name}_env_num 1 for this dataset."
        )
    

args = parse_args()
# make dirs

if not os.path.exists(args.save_video_dir):
    os.makedirs(args.save_video_dir)
if not os.path.exists(args.save_model_dir):
    os.makedirs(args.save_model_dir)
if not os.path.exists(args.save_control_dir):
    os.makedirs(args.save_control_dir)
if not os.path.exists(args.save_visualize_dir):
    os.makedirs(args.save_visualize_dir)
if not os.path.exists(args.test_save_dir):
    os.makedirs(args.test_save_dir)
    
now = int(round(time.time()*1000))
now = time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(now/1000))
save_video_dir = os.path.join(args.save_video_dir, now)
checkpoint_dir = os.path.join(args.save_model_dir, now)
final_model_path = os.path.join(args.save_model_dir, now + '.pth')
train_control_dir = os.path.join(args.save_control_dir, now)
test_control_dir = os.path.join(args.test_save_dir, now)
train_visualize_dir = os.path.join(args.save_visualize_dir, now)
for run_dir in (checkpoint_dir, train_control_dir, test_control_dir, train_visualize_dir):
    os.makedirs(run_dir, exist_ok=True)

video_episode_trigger = partial(
    should_record_video_episode,
    early_episodes=args.early_video_episodes,
    interval=args.video_episode_interval,
)

## load tool property: json file
print(f"load tool {args.which_tool} ...")
with open(args.tool_property_dir) as f:
    tp = json.load(f)

if args.which_tool == 'brush':
    tool = tools.Writing_Brush(tp["r_min"], tp["r_max"],
                                tp["l_min"], tp["l_max"],
                                tp["theta_min"], tp["theta_max"],
                                tp["theta_step"])
elif args.which_tool == 'ellipse':
    tool = tools.Ellipse(tp["r_min"], tp["r_max"],
                                tp["l_min"], tp["l_max"],
                                tp["theta_min"], tp["theta_max"],
                                tp["theta_step"])
    
elif args.which_tool == 'marker':
    tool = tools.Chisel_Tip_Marker(tp["r_min"], tp["r_max"],
                                tp["l_min"], tp["l_max"],
                                tp["theta_min"], tp["theta_max"],
                                tp["theta_step"])

validate_data_dir("train", args.train_data_dir)
validate_data_dir("test", args.test_data_dir)
train_img_num = utils.count_file_num(args.train_data_dir)
test_img_num = utils.count_file_num(args.test_data_dir)
validate_positive_int("--train_env_num", args.train_env_num)
validate_positive_int("--test_env_num", args.test_env_num)
validate_positive_int("--image_iter", args.image_iter)
validate_positive_int("--update", args.update)
validate_positive_int("--step_per_collect", args.step_per_collect)
validate_positive_int("--step_per_epoch", args.step_per_epoch)
validate_positive_int("--batch_size", args.batch_size)
validate_positive_int("--episode_per_test", args.episode_per_test)
validate_nonnegative_int("--early_video_episodes", args.early_video_episodes)
validate_positive_int("--video_episode_interval", args.video_episode_interval)
validate_positive_int("--checkpoint_interval", args.checkpoint_interval)
validate_nonnegative_int("--early_artifact_episodes", args.early_artifact_episodes)
validate_positive_int("--artifact_episode_interval", args.artifact_episode_interval)
validate_data_split("train", args.train_data_dir, train_img_num, args.train_env_num)
validate_data_split("test", args.test_data_dir, test_img_num, args.test_env_num)
print(f"train images: {train_img_num}, train envs: {args.train_env_num}")
print(f"test images: {test_img_num}, test envs: {args.test_env_num}")
print(
    f"video snapshots: every episode < {args.early_video_episodes}, "
    f"then every {args.video_episode_interval} episodes"
)
print(f"model checkpoints: {checkpoint_dir}, every {args.checkpoint_interval} epoch(s)")
print(
    f"control/visual snapshots: episodes 1-{args.early_artifact_episodes}, "
    f"then every {args.artifact_episode_interval} episodes"
)
env = gym.make('CalliEnv-v0',tool = tool,
                folder_path = args.train_data_dir,
                env_num = 1,
                env_rank=(0,train_img_num),
                render_mode = args.train_render_mode,
                output_path = train_control_dir,
                visualize_path = None,
                skel_datatype = args.skel_datatype,
                smooth_action_weight=args.smooth_action_weight,
                smooth_theta_weight=args.smooth_theta_weight,
                smooth_pos_weight=args.smooth_pos_weight,
                smooth_penalty_max=args.smooth_penalty_max,
                new_step_api = True)

print("make train envs...")
train_envs = SubprocVectorEnv([lambda i=i: RecordVideo(
                                            gym.make('CalliEnv-v0',tool = tool,
                                            folder_path = args.train_data_dir,
                                            output_path = train_control_dir,
                                            visualize_path = train_visualize_dir,
                                            env_num = args.train_env_num,
                                            env_rank=(int(train_img_num/args.train_env_num*i),train_img_num),
                                            render_mode = args.train_render_mode,
                                            image_iter = args.image_iter,
                                            start_update = args.start_update,
                                            update = args.update,
                                            ema_gamma = 0.9,
                                            skel_datatype = args.skel_datatype,
                                            smooth_action_weight=args.smooth_action_weight,
                                            smooth_theta_weight=args.smooth_theta_weight,
                                            smooth_pos_weight=args.smooth_pos_weight,
                                            smooth_penalty_max=args.smooth_penalty_max,
                                            early_artifact_episodes=args.early_artifact_episodes,
                                            artifact_episode_interval=args.artifact_episode_interval,
                                            new_step_api = True),
                                        video_folder= save_video_dir,
                                        episode_trigger=video_episode_trigger,
                                        name_prefix= 'trainvids_'+str(i),
                                        new_step_api=True) for i in range (args.train_env_num)])

print("make test envs...")
test_envs = DummyVectorEnv([lambda i=i: RecordVideo(
                                            gym.make('CalliEnv-v0',tool = tool,
                                            folder_path = args.test_data_dir,
                                            output_path = test_control_dir,
                                            visualize_path = None,
                                            env_num = args.test_env_num,
                                            env_rank=(int(test_img_num/args.test_env_num*i),test_img_num),
                                            render_mode = args.test_render_mode,
                                            image_iter = args.image_iter,
                                            start_update = args.start_update,
                                            update = args.update,
                                            ema_gamma = 0.9,
                                            skel_datatype = args.skel_datatype,
                                            smooth_action_weight=args.smooth_action_weight,
                                            smooth_theta_weight=args.smooth_theta_weight,
                                            smooth_pos_weight=args.smooth_pos_weight,
                                            smooth_penalty_max=args.smooth_penalty_max,
                                            early_artifact_episodes=args.early_artifact_episodes,
                                            artifact_episode_interval=args.artifact_episode_interval,
                                            new_step_api = True),
                                        video_folder= save_video_dir,
                                        episode_trigger=video_episode_trigger,
                                        name_prefix= 'testvids_'+str(i),
                                        new_step_api=True) for i in range (args.test_env_num)])


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
train_envs.seed(args.seed)
test_envs.seed(args.seed)


observe_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[0]

## optional Fourier/RBF state features
f_kwargs: Dict[str, Any] = {
                "sigma": args.sigma,
                "train_B": args.train_B,
                "fourier_dim": args.fourier_dim,
                "concatenate_fourier":args.concatenate_fourier,
                "rbf_dim": args.rbf_dim,
                "rbf_sigma": args.rbf_sigma,
                "train_rbf_centers": args.train_rbf_centers,
                "concatenate_rbf": args.concatenate_rbf,
            }

# actor_net = My_Siren(observe_shape, hidden_sizes=args.actor_network_shape,device=device, **f_kwargs).to(device)
# critic_net = My_Siren(observe_shape,action_shape,hidden_sizes=args.critic_network_shape,concat=True,device=device, **f_kwargs).to(device)

actor_net = My_MLP(observe_shape, hidden_sizes=args.actor_network_shape,device=device,\
                   learn_fourier=args.learn_fourier, learn_rbf=args.learn_rbf, **f_kwargs).to(device)
critic_net = My_MLP(observe_shape,action_shape,hidden_sizes=args.critic_network_shape,concat=True,\
                    device=device, learn_fourier=args.learn_fourier, learn_rbf=args.learn_rbf, **f_kwargs).to(device)

actor = ActorProb(actor_net, action_shape, args.mu_std_net, device=device).to(device)
critic_1 = Critic(critic_net, device=device).to(device)
critic_2 = Critic(critic_net, device=device).to(device)

if args.zero_init_actor:
    initialize_actor_near_zero(actor, args.initial_actor_std)

actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
critic1_optim = torch.optim.Adam(critic_1.parameters(), lr=args.critic_lr)
critic2_optim = torch.optim.Adam(critic_2.parameters(), lr=args.critic_lr)

## entropy
target_entropy = args.target_entropy_ratio * torch.log(torch.tensor(float(action_shape)))
log_alpha = torch.zeros(1, requires_grad=True, device=device)
alpha_optim = torch.optim.Adam([log_alpha], lr=args.actor_lr)
alpha = (target_entropy, log_alpha, alpha_optim)

## SAC policy
policy = SACPolicy(actor,
                   actor_optim,
                   critic_1,
                   critic1_optim,
                   critic_2,
                   critic2_optim,
                   tau = args.tau,
                   gamma = args.gamma,
                   alpha=alpha
                )

## replay_buffer
buf = VectorReplayBuffer(args.buffer_size, len(train_envs))

## collector
train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
initial_collect_steps = (2**9) * args.train_env_num
if args.initial_collect_mode == 'zero':
    train_collector.policy = ZeroActionPolicy(action_shape)
    train_collector.collect(n_episode=args.train_env_num)
    train_collector.policy = policy
elif args.initial_collect_mode == 'random':
    train_collector.collect(n_step=initial_collect_steps, random=True)
else:
    train_collector.collect(n_step=initial_collect_steps)
test_collector = Collector(policy, test_envs, exploration_noise=False)

## logger
log_path = os.path.join(args.logdir, now)
writer = SummaryWriter(log_path)
logger = TensorboardLogger(writer, save_interval=args.checkpoint_interval)


def save_initial_policy() -> None:
    torch.save(policy.state_dict(), os.path.join(checkpoint_dir, "epoch_0000_initial.pth"))


def save_best_policy(current_policy: BasePolicy) -> None:
    torch.save(current_policy.state_dict(), os.path.join(checkpoint_dir, "best.pth"))


def save_policy_checkpoint(epoch: int, env_step: int, gradient_step: int) -> str:
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"epoch_{epoch:04d}_envstep_{env_step:09d}.pth",
    )
    torch.save(policy.state_dict(), checkpoint_path)
    return checkpoint_path


save_initial_policy()
print("start training...")

## trainer
'''
step_per_collect: number of collect transitions before updating the neural network once.

episode_per_test: number of evaluaton episodes during each testing phase.

update_per_step: network update times on collecting 'step_per_collect' transitions.

train_fn: a hook to perform custom operations at the beginning of each epoch's training.

test_fn: a hook to perform custom operations at the beginning of each epoch's testing.

stop_fn: a function that receives the average undiscounted reward from the test results
         and returns a boolean value indicating whether the goal has been reached.
'''
step_per_collect = (args.step_per_collect * args.train_env_num)
assert step_per_collect % args.train_env_num == 0

result = offpolicy_trainer(policy=policy,
                        train_collector=train_collector,
                        test_collector=test_collector,
                        max_epoch=args.max_epoch,
                        step_per_epoch = args.step_per_epoch,  
                        step_per_collect= step_per_collect,
                        episode_per_test= args.episode_per_test,
                        batch_size=args.batch_size,
                        update_per_step = args.update_per_step,
                        resume_from_log= args.resume_from_log,
                        save_best_fn=save_best_policy,
                        save_checkpoint_fn=save_policy_checkpoint,
                        logger=logger,
                        verbose= True,
                        show_progress= True,
                        test_in_train= True
                        )
print(f'Finished training! Use {result["duration"]}')
torch.save(policy.state_dict(), final_model_path)
policy.eval()
collector = Collector(policy, test_envs, exploration_noise=True)
result = test_collector.collect(n_episode=1, render=1/30)

'''

python try_tianshou.py   \
    --train_data_dir data/train_data/   \
    --test_data_dir data/test_data/   \
    --train_env_num 6   \
    --test_env_num 6   \
    --which_tool brush   \
    --tool_property_dir tool_property/brush.json   \
    --logdir result/mlp_zero_init_log   \
    --save_video_dir result/mlp_zero_init_videos/   \
    --save_model_dir result/models/mlp_zero_init/   \
    --save_control_dir result/mlp_zero_init_arrays/   \
    --test_save_dir result/mlp_zero_init_test/   \
    --save_visualize_dir result/mlp_zero_init_vis/   \
    --initial_collect_mode zero   \
    --zero_init_actor   \
    --initial_actor_std 0.05   \
    --skel_datatype npy \
    --early_artifact_episodes 3 \
    --artifact_episode_interval 5 \
    --early_video_episodes 3 \
    --video_episode_interval 5 \
    --checkpoint_interval 2 \
    --smooth_action_weight 0 \
    --smooth_theta_weight 0.01 \
    --smooth_pos_weight 0 \
    --smooth_penalty_max 0.1

python try_tianshou.py \
  --train_data_dir data/train_data/ \
  --test_data_dir data/test_data/ \
  --train_env_num 6 \
  --test_env_num 6 \
  --which_tool brush \
  --tool_property_dir ./tool_property/brush.json \
  --logdir ./result/output.log \
  --learn_rbf \
  --rbf_dim 128 \
  --rbf_sigma 0.5 \
  --train_rbf_centers \
  --no-concatenate_rbf\
  --skel_datatype npy


python try_tianshou.py \
  --train_data_dir data/train_data/ \
  --test_data_dir data/test_data/ \
  --train_env_num 6 \
  --test_env_num 6 \
  --which_tool brush \
  --tool_property_dir tool_property/brush.json \
  --logdir result/output.log \
  --learn_rbf \
  --rbf_dim 64 \
  --rbf_sigma 0.8 \
  --train_rbf_centers \
  --concatenate_rbf \
  --skel_datatype npy

# Recommended brush run with smoothness reward and dense early snapshots.
# Run from rl_finetune/ so the relative paths below resolve correctly.
python try_tianshou.py \
  --train_data_dir ./data/train_data/ \
  --test_data_dir ./data/test_data/ \
  --skel_datatype npy \
  --train_env_num 1 \
  --test_env_num 1 \
  --which_tool brush \
  --tool_property_dir ./tool_property/brush.json \
  --train_render_mode rgb_array \
  --test_render_mode rgb_array \
  --save_video_dir ./result/mlp_zero_init_videos/ \
  --save_model_dir ./result/mlp_zero_init_models/ \
  --save_control_dir ./result/mlp_zero_init_arrays/ \
  --test_save_dir ./result/mlp_zero_init_test/ \
  --save_visualize_dir ./result/mlp_zero_init_vis/ \
  --logdir ./result/mlp_zero_init_log/ \
  --initial_collect_mode zero \
  --zero_init_actor \
  --initial_actor_std 0.05 \
  --smooth_action_weight 0.01 \
  --smooth_theta_weight 0.02 \
  --smooth_pos_weight 0.02 \
  --smooth_penalty_max 0.1 \
  --image_iter 10 \
  --start_update 1 \
  --update 1 \
  --early_video_episodes 20 \
  --video_episode_interval 2 \
  --checkpoint_interval 2 \
  --early_artifact_episodes 20 \
  --artifact_episode_interval 2 \
  --max_epoch 150 \
  --step_per_epoch 10000 \
  --step_per_collect 4 \
  --update_per_step 2 \
  --batch_size 2048 \
  --episode_per_test 10
'''
