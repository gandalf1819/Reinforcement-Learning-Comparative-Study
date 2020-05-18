# Reinforcement-Learning-Comparative-Study

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)  ![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen) ![Python](https://upload.wikimedia.org/wikipedia/commons/3/34/Blue_Python_3.6_Shield_Badge.svg)

Reinforcement learning is a type of machine learning that involves a software agent deciding to take an action in an environment leading to maximum cumulative rewards. The learner uses trial and error methods instead of already specified steps to discover actions leading to maximum rewards. This trial and error search and delayed rewards are the distinguishing features of reinforcement learning. The reinforcement learning algorithms are compared based on the score they obtain in solving the problem based on an environment. The time taken by algorithms on various maps corresponding to the environment is also compared in this paper. It can be observed that algorithm performance varies on different environments, but it is comparable to one another.

## Use of Deep Learning through Reinforcement Learning on Game Playing Agents

Deep Learning is enabling reinforcement learning to scale to problems that were previously intractable, such as learning to play video games directly from pixels. Deep reinforcement learning algorithms are also applied to robotics, allowing control policies for robots to be learned directly from camera inputs in the real world

The first, kickstarting the revolution in DRL, was the development of an algorithm that could learn to play a range of Atari 2600 video games at a superhuman level, directly from image pixels

In machine learning, the environment is typically formulated as a Markov decision process (MDP), as many reinforcement learning algorithms for this context utilize dynamic programming techniques.

## Reinforcement Learning

![alt text](https://www.kdnuggets.com/images/reinforcement-learning-fig1-700.jpg)

- Action (A): All the possible moves that the agent can take
- State (S): Current situation returned by the environment
- Reward (R): An immediate return send back from the environment to evaluate the last action
- Policy (π): The strategy that the agent employs to determine next action based on the current state
- Value (V): The expected long-term return with discount, as opposed to the short-term reward R. Vπ(s) is defined as the expected long-term return of the current state sunder policy π
- Q-value or action-value (Q): Q-value is similar to Value, except that it takes an extra parameter, the current action a. Qπ(s, a) refers to the long-term return of the current state s, taking action a under policy π.


## Markov Decision Process

RL can be described as a Markov decision process (MDP), which consists of:
- A set of states S, plus a distribution of starting states p(s0)
- A set of actions A
- Transition dynamics T (st+1|st, at) that map a state action pair at time t onto a distribution of states at time t + 1
- An immediate/instantaneous reward function R(st, at, st+1)
- A discount factor γ ∈ [0, 1], where lower values place more emphasis on immediate rewards

## Neural Networks

Neural Networks, or connectionist systems, are such who learn, i.e. progressively improve performance on, by considering examples, generally without task-specific programming. For example, in a task of identifying “cats”, a Neural Network learns to identify them without any prior knowledge about cats. Rather, it evolves its own set of characteristics from learning from the material they process. Commonly, Neural Networks are also called as “Global Approximator Functions”.

## Building Blocks of Neural Networks

The following are the three main layer labels of a Neural Network

1. Input Layer
2. Hidden Layer(s)
3. Output Layer

## Functions of a Neural Network

- Take inputs through the input layer
- Forward Propagation (Feed Forward)
- Performance Evaluation
- Backward Propagation
- Repeat to achieve a desired “Goal”

# Reinforcement Learning Algorithms

## 1. Q-learning

Can handle problems with stochastic transitions and rewards
Finds an optimal action-selection policy for any given finite Markov Decision Process

### Types:
- Policy Iteration: Starts with a random policy, then finds the value function of that policy, then finds the new policy based on previous value function.
- Value iteration: Starts with a random value function, then finds a new value function in an iterative process until it reaches optimal value function.

### Pros:
- Needs no accurate representation of the environment in order to be effective

### Cons:
- Lack of generality
- Higher computational  cost when larger state space

## 2. Deep Q Network

DQN is a form of Q-learning with function approximation using a neural network.
Tries to learn a state-action value function Q by minimizing temporal-difference errors.

### Pros:
- Relatively stable performance when bad gradient is estimated

### Cons:
- DQN fails if Q function (i.e. reward function) is too complex to be learned
- DQN has to go through an expensive action discretization process

## 3. Policy Gradient

Relies upon optimizing parameterized policies w.r.t expected return (long-term cumulative reward) by gradient descent.

### Pros:
- Applicable on wider range of problems, even where DQN fails
- Can be applied to model continuous action space since the policy network models probability distribution
- Capable of learning stochastic policies as it models probabilities of actions
- Usually show faster convergence rate than DQN

### Cons:
- Tendency to converge to a local optimal
-High variance in estimating the gradient, hence the estimation can be very noisy, and bad gradient estimate could adversely impact the stability of the learning algorithm

## 4. Actor Critic

- Combine Policy Gradients and Q-learning by training both an actor (the policy) and a critic (the Q-function).
- Actor decides which action to take, and the critic tells the actor how good its action was and how it should adjust
- Alleviates the task of the critic as it only has to learn the values of (state, action) pairs generated by the policy
- Can also incorporate Q-learning tricks e.g. experience replay
- Uses advantage function:  How much an action was better than expected

## Pre-Processing

The direct output of the Ping Pong game is 210x160x3. 
Each image is on a 0-255 color scale. 
- Processing:	
* Cropping  
* Removal of background layer
* Down-sampling

- For Policy Gradient, images of 80x80 size are generated 
- For Actor Critic, images of 80x80 size are generated 
- For Deep Q-Net, images of 84x84 size are generated  

## Actor-Critic
![actor-critic](https://user-images.githubusercontent.com/22028693/48115548-3c6f4480-e231-11e8-8a39-d04c2f042f45.png)

## Deep Q Network
![dqn](https://user-images.githubusercontent.com/22028693/48115588-67f22f00-e231-11e8-9d58-a7d0a90290ef.png)

## Implementation
![implementation](https://user-images.githubusercontent.com/22028693/48115607-793b3b80-e231-11e8-99a4-6f776f5dd7c8.png)

# Environment Creation for Android

Apart from comparing the currently available various Reinforcement Learning algorithms on unique environments, we also decided to create our own environment for the Android platform.

Currently, on the Android platform, there are few environments for games on which RL could be applied.
Key points about an environment:

At a given point of time, an environment, when probed, always returns the current observation of the screen, the immediate reward of the previous action performed, and whether the episode is finished or not.

A well optimized observation includes useful and contributing data for the agent to perceive and make decisions accordingly.

The reward is a number which is returned based in relation to the relative achievement of the defined goal. The reward has a lower cap as well as a higher cap. The lower cap, in most cases, means that the reward is negative and vice versa.

The “done” flag is a boolean value which is a trigger to reset the episode as no further action can be performed.

The environment allows the agent to perform actions inside it. Each unique environment has yet it’s own unique action space. The action space can be any large.

## What we did:

We captured continuous frames from the Android device through the ADB Interface taking the help of the Minicap library.

We pre-processed each frame before feeding it to the neural network so as to optimize the training time; we extracted useful data from the frames.

We defined the action space for the environment, for Subway Surfers for instance, which included defining simulated physical swipes, touches on the Android device.

We identified the “reward” value function to be used for the environment.

The “done” state is integrated with the pre-processing of the observation taken from the Android device’s frames.

We then implemented a Deep Q-Learning agent into our newly created environment.


## Steps:
- Get screen frames from device
* Used “minicap” framework for continuous frame capture from Android
- Pre-process frames
* Rescaled the frames to 320x200 shape
* Applied a grayscale on each frame
- Check for ‘Done’ state
* Used SSIM (Structural Similarity Index) to identify the “done” frame
- Decide action space and construct network
* Each game environment has its own action space; defined as all the possible actions the agent can perform
* Eg: Subway Surfers has 4 possible actions
* Eg: Flappy Bird has 2 possible actions
- Perform actions and get reward:
* Used “MonkeyRunner” and “AndroidViewClient” to send simulated physical touches to the device as agent inputs
* Depending on the “done” state, the reward is given; +1 for every action if not done state else 0

## Pre-processing
![flappy-bird-1](https://user-images.githubusercontent.com/22028693/48115696-c7503f00-e231-11e8-80a0-86113a76ff67.png)

![flappy-bird-2](https://user-images.githubusercontent.com/22028693/48115715-d505c480-e231-11e8-80eb-3b235402874b.png)

## Colored Frame Graphs
![colored-graphs-1](https://user-images.githubusercontent.com/22028693/48115646-9bcd5480-e231-11e8-860a-547f4b9d2096.png)

## Grayscale Frame Graphs
![grayscale](https://user-images.githubusercontent.com/22028693/48115677-b4d60580-e231-11e8-810a-48b7cc93db9f.png)

## Team

* [Nikhil Sangvikar]()
* [Madhujita Ambaskar](https://github.com/madhujita)
* [Chinmay Wyawahare](https://github.com/gandalf1819)
* [Prathmesh Palande](https://github.com/prathmeshpalande)

## Contributions

Please feel free to create a Pull Request for adding implementations of the algorithms discussed in different frameworks like TensorFlow, Keras, PyTorch, Caffe, etc. or improving the existing implementations.

## Support

If you found this useful, please consider starring(★) the repo so that it can reach a broader audience

## License

This project is licensed under the MIT License. Feel free to create a Pull Request for adding implementations or suggesting new ideas to make the analysis more insightful
