import argparse
import uuid
import os
import torch
import json
import numpy as np
import torch.backends.cudnn as cudnn
from utils import BatchAverageMeter
from torch.optim import Adam, lr_scheduler, RMSprop
from torch.utils.data import DataLoader
from ActorCriticNetwork import ActorCriticNetwork
from DataGenerator import TSPDataset
from TSPEnvironment import TSPInstanceEnv, VecEnv
import colorama
from tqdm import tqdm


def argparser():
    parser = argparse.ArgumentParser()

    # ----------------------------------- Data ---------------------------------- #
    parser.add_argument('--train_size',
                        default=5120, type=int, help='Training data size')
    parser.add_argument('--test_size',
                        default=256, type=int, help='Test data size')
    parser.add_argument('--test_from_data',
                        default=False, action='store_true', help='Test data size')
    parser.add_argument('--batch_size',
                        default=512, type=int, help='Batch size')
    parser.add_argument('--n_points',
                        type=int, default=20, help='Number of points in the TSP')

    # ---------------------------------- Train ---------------------------------- #
    parser.add_argument('--n_steps',
                        default=200,
                        type=int, help='Number of steps in each episode')
    parser.add_argument('--n',
                        default=30,
                        type=int, help='Number of steps to bootstrap')
    parser.add_argument('--gamma',
                        default=0.99,
                        type=float, help='Discount factor for rewards')
    parser.add_argument('--render',
                        default=False,
                        action='store_true', help='Render')
    parser.add_argument('--render_from_epoch',
                        default=0,
                        type=int, help='Epoch to start rendering')
    parser.add_argument('--update_value',
                        default=True,
                        action='store_true',
                        help='Use the value function for TD updates')
    parser.add_argument('--epochs',
                        default=200, type=int, help='Number of epochs')
    parser.add_argument('--lr',
                        type=float, default=0.001, help='Learning rate')
    parser.add_argument('--wd',
                        default=1e-5,
                        type=float, help='Weight decay')
    parser.add_argument('--beta',
                        type=float, default=0.005, help='Entropy loss weight')
    parser.add_argument('--zeta',
                        type=float, default=0.5, help='Value loss weight')
    parser.add_argument('--max_grad_norm',
                        type=float, default=0.3, help='Maximum gradient norm')
    parser.add_argument('--no_norm_return',
                        default=False,
                        action='store_true', help='Disable normalised returns')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 1)')
    parser.add_argument('--rms_prop',
                        default=False,
                        action='store_true', help='Disable normalised returns')
    parser.add_argument('--adam_beta1',
                        type=float, default=0.9, help='ADAM beta 1')
    parser.add_argument('--adam_beta2',
                        type=float, default=0.999, help='ADAM beta 2')
    # ----------------------------------- GPU ----------------------------------- #
    parser.add_argument('--gpu',
                        default=True, action='store_true', help='Enable gpu')
    parser.add_argument('--gpu_n',
                        default=0, type=int, help='Choose GPU')
    # --------------------------------- Network --------------------------------- #
    parser.add_argument('--input_dim',
                        type=int, default=2, help='Input size')
    parser.add_argument('--embedding_dim',
                        type=int, default=128, help='Embedding size')
    parser.add_argument('--hidden_dim',
                        type=int, default=128, help='Number of hidden units')
    parser.add_argument('--n_rnn_layers',
                        type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--n_actions',
                        type=int, default=2, help='Number of nodes to output')
    parser.add_argument('--graph_ref',
                        default=False,
                        action='store_true',
                        help='Use message passing as reference')

    # ----------------------------------- Misc ---------------------------------- #
    parser.add_argument("--name", type=str, default="", help="Name of the run")
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--watch_grad', default=False, action='store_true', help='Record grad statstics while training')
    parser.add_argument('--tensorboard', default=True, action='store_true', help='use tensorboard')

    return parser

# ANCHOR initialize
# *-------------------------------- initialize --------------------------------* #

# create {} to log stuff
log = {}
log['hyperparameters'] = {}
args = argparser().parse_args()

# log hyperparameters
for arg in vars(args):
    log['hyperparameters'][arg] = getattr(args, arg)

# give it a clever name :D
# unique id in case of no name given
id = args.name or uuid.uuid4().hex
print("Name:", str(id))

# select a gpu to use
if args.gpu and torch.cuda.is_available():
    USE_CUDA = True
    print('Using GPU, {} devices available.'.format(
        torch.cuda.device_count()))
    torch.cuda.set_device(args.gpu_n)
    print("GPU: %s" % torch.cuda.get_device_name(
        torch.cuda.current_device()))
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    USE_CUDA = False
    device = torch.device("cpu")
torch.autograd.set_detect_anomaly(True)

# * Initiate the logs *
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(comment="-pg_" + id)

bam = BatchAverageMeter(['train_policy_loss','train_entropy_loss','train_value_loss','train_loss',
    'train_reward','train_init_dist','train_best_dist','val_reward','val_init_dist','val_best_dist'])
epoch_bam = BatchAverageMeter(['train_policy_loss','train_entropy_loss','train_value_loss','train_loss'])

best_running_reward = 0
val_best_dist = 1e10
best_gap = 1e10
count_learn = 0


# ANCHOR supporting functions
# *--------------------------- supporting functions ---------------------------* #
def printlog(epoch,prefix='',suffix='',**params):
    print(prefix, 'epoch', epoch, end=" - ")
    print(*(f'{k.replace("_"," ")}: {v:.3f}' for k,v in params.items()),sep=' | ', end=suffix+'\n')


def batchwrite_scalar(step, **params):
    if args.tensorboard:
        for k,v in params.items():
            writer.add_scalar(k, v, step)


class Buffer:
    def __init__(self):
        # action & reward buffer
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.entropies = []

    def clear_buffer(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.values[:]
        del self.entropies[:]


def select_action(policy, state, hidden, buffer, best_state):
    probs, action, log_probs_action, v, entropy, hidden = policy(
        state, best_state, hidden)
    buffer.log_probs.append(log_probs_action)
    buffer.states.append(state)
    buffer.actions.append(action)
    buffer.values.append(v)
    buffer.entropies.append(entropy)
    return action, v, hidden


# ANCHOR training and validation process
# *---------------------- training and validation process ---------------------* #
def learn(R, t_s, beta, zeta, count_learn, epoch, buffer):
    """
    Training. Calcultes actor and critic losses and performs backprop.
    """
    # Starting sum of losses for logging
    if t_s == 0:
        epoch_bam.reset()

    # Returns
    if R is None:
        R = torch.zeros((args.batch_size, 1))
    returns = []  # returns for each state discounted
    for s in range(len(buffer.rewards)-1, -1, -1):
        R = buffer.rewards[s] + args.gamma * R
        returns.insert(0, R)

    returns = torch.stack(returns).detach()
    if not args.no_norm_return:
        r_mean = returns.mean()
        r_std = returns.std()
        eps = np.finfo(np.float32).eps.item()  # small number to avoid div/0
        returns = (returns - r_mean)/(r_std + eps)

    # num of experiences in this "batch" of experiences
    n_experiences = args.batch_size*args.n
    # transform lists to tensor
    values = torch.stack(buffer.values)
    log_probs = torch.stack(buffer.log_probs).mean(2).unsqueeze(2)
    entropies = torch.stack(buffer.entropies).mean(2).unsqueeze(2)
    advantages = returns - values
    p_loss = (-log_probs*advantages.detach()).mean()
    v_loss = zeta*(returns - values).pow(2).mean()
    e_loss = (0.9**(epoch+1))*beta*entropies.sum(0).mean()

    optimizer.zero_grad()

    p_loss.backward(retain_graph=True)
    r_loss = - e_loss + v_loss

    r_loss.backward()
    # nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
    optimizer.step()
    loss = p_loss + r_loss

    # track statistics
    batchwrite_scalar(count_learn, 
        Returns=returns.mean(), 
        Advantage=advantages.mean(), 
        Loss_Actor=p_loss, 
        Loss_Critic=v_loss, 
        Loss_Entropy=e_loss, 
        Loss_Total=loss)

    if args.watch_grad:
        grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                            for p in policy.parameters()
                            if p.grad is not None])
        batchwrite_scalar(count_learn,
            Gradients_L2=np.sqrt(np.mean(np.square(grads))),
            Gradients_Max=np.max(np.abs(grads)),
            Gradients_Var=np.var(grads))

    epoch_bam.batchupdate(n=n_experiences,
        train_policy_loss=p_loss.item(),
        train_entropy_loss=e_loss.item()/args.n,
        train_value_loss=v_loss.item(),
        train_loss=loss.item())

    buffer.clear_buffer()


def training(policy, dataloader):
    # ANCHOR training
    # save metrics for all batches
    global count_learn

    epoch_rewards = []
    epoch_initial_distances = []
    epoch_best_distances = []
    buffer = Buffer()

    for batch_idx, batch_sample in enumerate(dataloader):
        t = 0
        b_sample: torch.Tensor = batch_sample.to(device)
        batch_reward = 0

        # every batch defines a set of agents running the same policy
        env = VecEnv(TSPInstanceEnv, b_sample.shape[0], args.n_points)
        state, initial_distance, best_state = env.reset(b_sample)
        hidden = None

        while t < args.n_steps:
            t_s = t
            while t - t_s < args.n and t != args.n_steps:
                if args.render and epoch > args.render_from_epoch:
                    env.render()
                action, v, _ = select_action(policy, state, hidden, buffer, best_state)

                next_state, reward, _, best_distance, _, next_best_state = \
                    env.step(action.cpu().numpy())

                buffer.rewards.append(reward)
                batch_reward += reward

                state = next_state
                best_state = next_best_state
                t += 1
            if args.update_value:
                _, _, _, next_v, _, _ = policy(next_state, next_best_state, hidden)
                R = next_v
            else:
                R = None
            count_learn += 1
            learn(R, t_s, args.beta, args.zeta, count_learn, epoch, buffer)

        epoch_rewards.append(batch_reward)
        epoch_best_distances.append(best_distance)
        epoch_initial_distances.append(initial_distance)

    # logging
    epoch_reward = sum(epoch_rewards).mean().item()/len(epoch_rewards)
    epoch_initial_distance = sum(epoch_initial_distances).mean().item()/len(epoch_initial_distances)
    epoch_best_distance = sum(epoch_best_distances).mean().item()/len(epoch_initial_distances)

    bam.batchupdate(
        train_policy_loss=epoch_bam['train_policy_loss'].avg, 
        train_entropy_loss=epoch_bam['train_entropy_loss'].avg, 
        train_value_loss=epoch_bam['train_value_loss'].avg, 
        train_loss=epoch_bam['train_loss'].avg,
        train_reward=epoch_reward,
        train_init_dist=epoch_initial_distance,
        train_best_dist=epoch_best_distance)
    batchwrite_scalar(epoch, 
        Rewards_Training=epoch_reward,
        Tour_Cost_Training=bam['train_best_dist'].val/10000)


def validation(policy, dataloader):
    # ANCHOR validation

    val_epoch_rewards = []
    val_epoch_best_distances = []
    val_epoch_initial_distances = []
    sum_probs = 0
    for val_batch_idx, val_batch_sample in enumerate(dataloader):
        val_b_sample = val_batch_sample.to(device)
        val_batch_reward = 0
        env = VecEnv(TSPInstanceEnv, val_b_sample.shape[0], args.n_points)
        state, initial_distance, best_state = env.reset(val_b_sample)
        t = 0
        hidden = None
        while t < args.n_steps:
            with torch.no_grad():
                probs, action, _, _, _, _ = policy(state, best_state, hidden)
            sum_probs += probs
            action = action.cpu().numpy()
            state, reward, _, best_distance, distance, best_state = env.step(action)
            val_batch_reward += reward
            t += 1

        val_epoch_rewards.append(val_batch_reward)
        val_epoch_best_distances.append(best_distance)
        val_epoch_initial_distances.append(initial_distance)

    avg_probs = torch.sum(sum_probs, dim=0) / (args.n_steps*args.test_size)*100
    avg_probs = avg_probs.cpu().numpy().round(2)

    # logging
    val_epoch_reward = sum(val_epoch_rewards).mean().item()/len(val_epoch_rewards)
    val_epoch_best_distance = sum(val_epoch_best_distances).mean().item()/len(val_epoch_rewards)
    val_epoch_initial_distance = sum(val_epoch_initial_distances).mean().item()/len(val_epoch_rewards)

    bam.batchupdate(
        val_reward=val_epoch_reward,
        val_init_dist=val_epoch_initial_distance,
        val_best_dist=val_epoch_best_distance)
    batchwrite_scalar(epoch,
        Rewards_Testing=val_epoch_reward,
        Tour_Cost_Testing=bam['val_best_dist'].val/10000)


# ANCHOR main program
# *------------------------------- main program -------------------------------* #

# * load the model *
policy = ActorCriticNetwork(
    args.input_dim, args.embedding_dim, args.hidden_dim,
    args.n_points, args.n_rnn_layers, args.n_actions, args.graph_ref
)

if args.load_path != '':
    print('  [*] Loading model from {}'.format(args.load_path))
    policy.load_state_dict(torch.load(os.path.join(os.getcwd(), args.load_path))['policy'], strict=False)

if USE_CUDA:
    policy.cuda()
    # policy = torch.nn.DataParallel(
    #     policy, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

# * define the optimizer and scheduler *
if args.rms_prop:
    optimizer = RMSprop(policy.parameters(), lr=args.lr, weight_decay=args.wd)
else:
    optimizer = Adam(policy.parameters(),
                        lr=args.lr,
                        weight_decay=args.wd,
                        betas=(args.adam_beta1, args.adam_beta2))
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)

# * load the test data *
if args.test_from_data:
    test_data = TSPDataset(dataset_fname=os.path.join(args.data_dir, f'TSP{args.n_points}-data.json'),
                            num_samples=args.test_size)
else:
    test_data = TSPDataset(dataset_fname=None, size=args.n_points, num_samples=args.test_size)

test_loader = DataLoader(test_data, batch_size=args.test_size, shuffle=False)


# ANCHOR mainloop
for epoch in range(args.epochs):
    # randomly generated points
    train_data = TSPDataset(dataset_fname=None, size=args.n_points, num_samples=args.train_size)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

    training(policy=policy, dataloader=train_loader)
    validation(policy=policy, dataloader=test_loader)

    scheduler.step()

    if args.test_from_data:
        gap = ((bam['val_best_dist'].val/10000) / np.mean(test_data.opt) - 1.0)*100
        batchwrite_scalar(epoch, Gap_Testing=gap)
        
    if bam['val_reward'].exp_avg > best_running_reward \
            or bam['val_best_dist'].val < val_best_dist \
            or (args.test_from_data and gap < best_gap):

        print('\033[1;37;40m Saving model...\033[0m')
        model_dir = os.path.join(args.model_dir, str(id))
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        checkpoint = {
            'policy': policy.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(model_dir,
                                            'pg-{}-TSP{}-epoch-{}.pt'
                                            .format(str(id), args.n_points, epoch)))
        torch.save(policy, os.path.join(model_dir,
                                        'full-model-pg-{}-TSP{}-epoch-{}.pt'
                                        .format(str(id), args.n_points, epoch)))
        best_running_reward = bam['val_reward'].exp_avg
        val_best_dist = bam['val_best_dist'].val
        if args.test_from_data:
            best_gap = gap

    if epoch % args.log_interval == 0:
        bam.logto(log)

        printlog(
            prefix=colorama.Fore.GREEN+'Train -', suffix=colorama.Fore.RESET, 
            epoch=epoch, rwd=bam['train_reward'].val, running_rwd=bam['train_reward'].exp_avg, 
            best_cost=bam['train_best_dist'].val/10000)
        printlog(
            prefix=colorama.Fore.YELLOW+"Valid -", suffix=colorama.Fore.RESET,
            epoch=epoch, rwd=bam['val_reward'].val, running_rwd=bam['val_reward'].exp_avg, 
            best_cost=bam['val_best_dist'].val/10000,
            **({"optimal_cost": np.mean(test_data.opt), "gap": gap} if args.test_from_data else {}))

        with open(os.path.join(args.log_dir,'pg-{}-TSP{}.json'.format(str(id), args.n_points)), 'w') as outfile:
            json.dump(log, outfile, indent=4)

print("done.")