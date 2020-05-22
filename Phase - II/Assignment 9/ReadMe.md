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


<img src="https://github.com/vinayakumarvs/EVA/blob/master/Phase%20-%20II/Assignment%209/Images/Q-Learning.png" width="50%" height="50%">
