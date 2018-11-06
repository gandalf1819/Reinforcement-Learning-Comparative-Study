# Reinforcement-Learning-Comparative-Study

Abstract—Reinforcement learning is a type of machine learning that involves a software agent deciding to take an action in an environment leading to maximum cumulative rewards. The learner uses trial and error methods instead of already specified steps to discover actions leading to maximum rewards. This trial and error search and delayed rewards are the distinguishing features of reinforcement learning. The reinforcement learning algorithms are compared based on the score they obtain in solving the problem based on an environment. The time taken by algorithms on various maps corresponding to the environment is also compared in this paper. It can be observed that algorithm performance varies on different environments, but it is comparable to one another.

Apart from comparing the currently available various Reinforcement Learning algorithms on unique environments, we also decided to create our own environment for the Android platform.

Currently, on the Android platform, there are few environments for games on which RL could be applied.
Key points about an environment:

At a given point of time, an environment, when probed, always returns the current observation of the screen, the immediate reward of the previous action performed, and whether the episode is finished or not.

A well optimized observation includes useful and contributing data for the agent to perceive and make decisions accordingly.

The reward is a number which is returned based in relation to the relative achievement of the defined goal. The reward has a lower cap as well as a higher cap. The lower cap, in most cases, means that the reward is negative and vice versa.

The “done” flag is a boolean value which is a trigger to reset the episode as no further action can be performed.

The environment allows the agent to perform actions inside it. Each unique environment has yet it’s own unique action space. The action space can be any large.

What we did:

We captured continuous frames from the Android device through the ADB Interface taking the help of the Minicap library.

We pre-processed each frame before feeding it to the neural network so as to optimize the training time; we extracted useful data from the frames.

We defined the action space for the environment, for Subway Surfers for instance, which included defining simulated physical swipes, touches on the Android device.

We identified the “reward” value function to be used for the environment.

The “done” state is integrated with the pre-processing of the observation taken from the Android device’s frames.

We then implemented a Deep Q-Learning agent into our newly created environment.

![alt text](https://www.kdnuggets.com/images/reinforcement-learning-fig1-700.jpg)
