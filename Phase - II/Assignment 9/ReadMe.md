## T3D OR TWIN DELAYED DDPG ##

#### What is Reinforcement Learning?

Reinforcement Learning(RL) is a type of machine learning technique that enables an agent to learn in an interactive environment by trial and error using feedback from its own actions and experiences.

###### Some key terms that describe the basic elements of an RL problem are:
1.	Environment — Physical world in which the agent operates
2.	State — Current situation of the agent
3.	Reward — Feedback from the environment
4.	Policy — Method to map agent’s state to actions
5.	Value — Future reward that an agent would receive by taking an action in a particular state


Markov Decision Processes(MDPs) are mathematical frameworks to describe an environment in RL and almost all RL problems can be formulated using MDPs. An MDP consists of a set of finite environment states S, a set of possible actions A(s) in each state, a real valued reward function R(s) and a transition model P(s’, s | a). However, real world environments are more likely to lack any prior knowledge of environment dynamics. Model-free RL methods come handy in such cases.

Q-learning is a commonly used model-free approach which can be used for building a self-playing agent. It revolves around the notion of updating Q values which denotes value of performing action a in state s. The following value update rule is the core of the Q-learning algorithm.


<img src="https://github.com/vinayakumarvs/EVA/blob/master/Phase%20-%20II/Assignment%209/Images/Q-Learning.png" width="60%" height="50%">

#### What are some of the most used Reinforcement Learning algorithms?

Q-learning and SARSA (State-Action-Reward-State-Action) are two commonly used model-free RL algorithms. They differ in terms of their exploration strategies while their exploitation strategies are similar. While Q-learning is an off-policy method in which the agent learns the value based on action a* derived from the another policy, SARSA is an on-policy method where it learns the value based on its current action a derived from its current policy. These two methods are simple to implement but lack generality as they do not have the ability to estimates values for unseen states.

This can be overcome by more advanced algorithms such as Deep Q-Networks(DQNs) which use Neural Networks to estimate Q-values. But DQNs can only handle discrete, low-dimensional action spaces.

Deep Deterministic Policy Gradient(DDPG) is a model-free, off-policy, actor-critic algorithm that tackles this problem by learning policies in high dimensional, continuous action spaces. The figure below is a representation of actor-critic architecture.

###### Let’s understand the T3D step by step :

##### ***1.*** Import Required Libraries. Few important Libraries are
    a. https://pytorch.org: We use PyTorch for our neural network implementation
    
    b. Gym: This provides a variety of environments like Atari, MuJoCo, etc for our reinforcement learning experiments
    
    c. https://github.com/benelot/pybullet-gym: Library providing physics based environment for our experiment
    
##### ***2.*** Initialize the Experience Replay Memory, with a fixed size (i.e. 1000000 in this exercise). We will populate it with each new transition

An experience (aka transition) is defined by the following:

    a. s: current state of an agent
    b. a: agent’s next action to go to next state
    c. s': new state of an agent reaches after taking an action (a)
    d. r: reward an agent receive for going from state (s) to state (s') by taking action (a)

Agent playing in an environment randomly and the experience of these movements stored in replay buffer memory. While training these batch of experiences sampled to train an agent. The memory is being replaced with the new experiences and deletes the old one to accommodate in a specified memory.


<img src="https://github.com/vinayakumarvs/EVA/blob/master/Phase%20-%20II/Assignment%209/Images/Step1.png" width="60%" height="50%">


    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, \
                batch_dones = [], [], [], [], []
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy = False))
            batch_next_states.append(np.array(next_state, copy = False))
            batch_actions.append(np.array(action, copy = False))
            batch_rewards.append(np.array(reward, copy = False))
            batch_dones.append(np.array(done, copy = False))
        return np.array(batch_states), np.array(batch_next_states), \
                np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), \
                np.array(batch_dones).reshape(-1, 1)
