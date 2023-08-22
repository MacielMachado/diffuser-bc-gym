import cv2
import gym
import torch
import wandb
import numpy as np
from scipy.stats import logistic
from cart_racing_v2 import CarRacing
from data_preprocessing import DataHandler
from record_observations import RecordObservations
from models import Model_Cond_Diffusion, Model_cnn_mlp, Model_no_cnn


class Tester(RecordObservations):
    def __init__(self, model, env, render=True, device="cpu"):
        super(Tester).__init__()
        self.model = model
        self.env = env
        self.render = render
        self.device = device

    def run(self, run_wandb, name='', random=False):
        if run_wandb:
            self.config_wandb(project_name="cart_pole_bc", name=name)
        obs, _ = self.env.reset()
        reward = 0
        counter=0
        done = False
        truncated = False
        while counter < 1000:
            self.model.eval()
            obs_tensor = obs
            torch.from_numpy(obs_tensor).float().to(self.device).shape
            obs_tensor = (
                torch.Tensor(obs_tensor).type(torch.FloatTensor).to(self.device)
                )
            action = int(np.round(logistic.cdf(self.model.sample(obs_tensor).to(self.device).detach().cpu().numpy()[0]))[0])
            if random:
                action = self.env.action_space.sample()
            obs, new_reward, done, _, truncated = self.env.step(action)
            reward += new_reward
            counter += 1
            print(f'reward: {reward} | counter: {counter}')
            if done or truncated: 
                break
            if run_wandb:
                wandb.log({"reward": reward})
        if run_wandb: wandb.finish()

    def config_wandb(self, project_name, name):
        config={}
        if name != '':
            return wandb.init(project=project_name, name=name, config=config)
        return wandb.init(project=project_name, config=config)

if __name__ == '__main__':
    n_epoch = 40
    lrate = 1e-4
    device = "cpu"
    n_hidden = 128
    batch_size = 32
    n_T = 50
    net_type = "fc"
    drop_prob = 0.0
    extra_diffusion_steps = 16
    x_shape = 4
    y_dim = 1

    env = gym.make('CartPole-v1', render_mode="human")
    nn_model = Model_no_cnn(
        x_shape, n_hidden, y_dim, embed_dim=128, net_type=net_type
    ).to(device)

    model = Model_Cond_Diffusion(
        nn_model,
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        x_dim=x_shape,
        y_dim=y_dim,
        drop_prob=drop_prob,
    guide_w=0.0,)

    # model.load_state_dict(torch.load("model_casa2.pkl"))
    model.load_state_dict(torch.load("model_novo_bc.pkl"))

    stop = 1
    tester = Tester(model, env, render=True)
    tester.run(run_wandb=False, random=True)