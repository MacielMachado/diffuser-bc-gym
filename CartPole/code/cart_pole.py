from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import numpy as np
import time
import gym
import os

class CartPoleFuzzy():
    '''
    '''
    def __init__(self, random):
        '''
        '''
        self.random = random
        self.actions = []
        self.observations = []
        self.path = os.getcwd() + '/generate_dataset/' + 'CartPole/'

    def run(self, save, render):
        '''
        '''
        self._create_env()
        self._game_loop(save, render)

    def _create_env(self):
        '''
        '''
        self.env = gym.make('CartPole-v1', render_mode="human")

    def _game_loop(self, save, render):
        '''
        '''
        self._create_fuzzy_agent()
        obs = self.env.reset()
        obs, rew, done, _, _ = self.env.step(0)
        rew_sum = 0
        self.reward_list = []
        counter = 0
        while counter < 1000:
            if render: self.env.render()

            self.agent.input['theta']=obs[2]
            self.agent.input['omega']=obs[3]
            self.agent.compute()
            action = self.agent.output['action']
            action = 1 if action >= 1 else 0
            if self.random:
                action = self.env.action_space.sample()
            self.save_game(obs, action)
            obs, rew, done, _, _ = self.env.step(action)
            rew_sum += rew
            counter += 1
            print(f"counter: {counter}")
            # print(f'pos: {obs[0]}, vel: {obs[1]}, angle: {obs[2]}, ang vel: {obs[3]}, rew: {rew_sum}')
            if done: 
                self.env.reset()
                time.sleep(2)
                self.reward_list.append(rew_sum)
                counter += 1
                print(counter)
                rew_sum = 0

        if save:
            hash = str(int(time.time()))
            np.save(self.path+'states'+'.npy', self.observations)
            np.save(self.path+'actions'+'.npy', self.actions)
            
        self._plot_reward()
        self.env.reset()
        self.env.close()

    def _plot_reward(self):
        '''
        '''
        plt.plot(self.reward_list)
        plt.ylim([0, 600])
        plt.ylabel('Reward')
        plt.xlabel('Episode')
        plt.title('CartPole Rewards')
        plt.show()

    def _create_fuzzy_agent(self):
        '''
        '''
        self._fuzzy_antecedent()
        self._fuzzy_consequent()
        self._membership_functions()
        self._rules()
        self._control_system()

    def _fuzzy_antecedent(self):
        '''
        '''
        self.theta_antecedent = ctrl.Antecedent(np.arange(-0.420, 0.420, 0.005), 'theta')
        self.omega_antecedent = ctrl.Antecedent(np.arange(-5.5, 5.5, 0.001), 'omega')

    def _fuzzy_consequent(self):
        '''
        '''
        self.action_consequent = ctrl.Consequent(np.arange(0, 2, 0.001), 'action')

    def _membership_functions(self):
        '''
        '''
        self.theta_antecedent.automf(names=['theta_neg_alto', 'theta_neg_baixo',
                                    'theta_pos_baixo', 'theta_pos_alto'])
        self.omega_antecedent.automf(names=['omega_neg', 'omega_pos'])
        self.action_consequent.automf(names=['action_0', 'action_1'])
        
        # Theta
        self.theta_antecedent['theta_neg_alto']=fuzz.trapmf(self.theta_antecedent.universe,
                                           [-0.42, -0.42, -0.03, -0.02])
        self.theta_antecedent['theta_neg_baixo']=fuzz.trapmf(self.theta_antecedent.universe,
                                                [-0.03, -0.02, 0, 0])
        self.theta_antecedent['theta_pos_baixo']=fuzz.trapmf(self.theta_antecedent.universe,
                                                [0, 0, 0.02, 0.03])
        self.theta_antecedent['theta_pos_alto']=fuzz.trapmf(self.theta_antecedent.universe,
                                                [0.02, 0.03, 0.42, 0.42])

        # Omega
        self.omega_antecedent['omega_neg']=fuzz.trapmf(self.omega_antecedent.universe,
                                                [-5.5, -0.5, 0, 0])
        self.omega_antecedent['omega_pos']=fuzz.trapmf(self.omega_antecedent.universe,
                                                [0, 0, 0.5, 5.5])

        # Action
        self.action_consequent['action_0']=fuzz.trapmf(self.action_consequent.universe, [0, 0, 0.5, 1])
        self.action_consequent['action_1']=fuzz.trapmf(self.action_consequent.universe, [1, 1.5, 2, 2])

    def _rules(self):
        '''
        '''
        rule1 = ctrl.Rule((self.theta_antecedent['theta_neg_baixo'] |
                           self.theta_antecedent['theta_pos_baixo']) &
                           self.omega_antecedent['omega_neg'],
                           self.action_consequent['action_0'])

        rule2 = ctrl.Rule((self.theta_antecedent['theta_neg_baixo'] |
                           self.theta_antecedent['theta_pos_baixo']) &
                           self.omega_antecedent['omega_pos'],
                           self.action_consequent['action_1'])

        rule3 = ctrl.Rule(self.theta_antecedent['theta_neg_alto'],
                           self.action_consequent['action_0'])

        rule4 = ctrl.Rule(self.theta_antecedent['theta_pos_alto'],
                           self.action_consequent['action_1'])
        
        self.rules = [rule1, rule2, rule3, rule4]

    def _control_system(self):
        '''
        '''
        ctrl_system=ctrl.ControlSystem(self.rules)
        self.agent = ctrl.ControlSystemSimulation(ctrl_system)

    def save_game(self, obs, action):
        '''
        '''
        os.makedirs(self.path, exist_ok=True)
        self.actions.append(action)
        self.observations.append(obs)


if __name__ == '__main__':
    CartPoleFuzzy(random=False).run(save=True, render=True)