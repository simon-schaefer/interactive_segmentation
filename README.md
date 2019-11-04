# Interactive Segmentation using Object Singulation
An accurate segmentation is crucial for performing manipulation tasks in real-world environments, but especially in cluttered environments is still poses a problem as adjacent objects are segmented together. This problem can be solved by enhancing visual with interactive segmentation, using push movements to singulate objects from each other. Current learning based approaches for this task use Deep Q-Learning (DQL) with a discretized action space. As using continuous actions instead would allow more complex and more efficient pushing movements, in this paper it was evaluated whether Proximal Policy Optimization (PPO) can also solve the task of object singulation. It is shown that with PPO only simple static scenes can be solved with the proposed setup, while in contrast also simple dynamic scenes could be solved using DQL.  

## Installing
The project is implemented in Python3 and based on the PyBullet physical simulation engine as well as on OpenAI baselines. In order to install these and other requirements as well as setup the project, run:

```
git clone https://github.com/ethz-asl/interactive_segmentation
cd interactive_segmentation
source scripts/setup.bash
```

## Training and Evaluation
To train a new policy the environment, the task (action, state, reward), the training algorithm (hyperparameters) as well as the logging behaviour can be configured in the config simulation.yaml file. Next to PPO also DeepQLearning is available. For training run the following command:

```
python3 train.py
```

For debugging and visualization purposes the "debug" and "visualize" flag can be set in the configuration file. Each training will create an output directory containing logging and debug files (such as the reward or success rate curve) as well as the trained model.

## Running an Example

Each trained policy can be evaluated, in its trained environment and task configuration.

```
python3 eval.py --policy=random2_qlearn_80 --visualize
```

Here the "random2_qlearn_80" policy is loaded and evaluted, which can be found in the policies directory. For evaluating other trained models use "--out=[directory]" instead of "--policy". 

## Built With

* [PyBullet](https://pybullet.org/wordpress/) - PyBullet engine  
* [OpenAI Baselines](https://openai.com/) - OpenAI Baselines RL algorithms



<!--
# Parameter Grid Search
| algo  | scene  | sensor    | rew_schedule | minibatch | network     | train_exploration_fraction |   | reward | success rate |
|-------|--------|-----------|--------------|-----------|-------------|----------------------------|---|--------|--------------|
|       |        |           |              |           |             |                            |   |        |              |
| ppo2  | fixed  | segmented | 0            | 8         | fc_small    |                            |   | >38    | 0.0          |
| ppo2  | fixed  | distances | 0            | 8         | fc_small    |                            |   | 20     | 0.0          |
| ppo2  | fixed  | segmented | 1            | 8         | fc_small    |                            |   | >28    | 0.0          |
| ppo2  | fixed  | distances | 1            | 8         | fc_small    |                            |   | 38     | 0.001        |
| ppo2  | fixed  | segmented | 0            | 128       | fc_small    |                            |   | ~940   | ~0.31        |
| ppo2  | fixed  | distances | 0            | 128       | fc_small    |                            |   | >1752  | >0.82        |
| ppo2  | fixed  | segmented | 1            | 128       | fc_small    |                            |   | >175   | 0.0          |
| ppo2  | fixed  | distances | 1            | 128       | fc_small    |                            |   | ~1220  | >0.45        |
| ppo2  | random | segmented | 0            | 8         | fc_small    |                            |   | >70    | 0.003        |
| ppo2  | random | distances | 0            | 8         | fc_small    |                            |   | >70    | 0.004        |
| ppo2  | random | segmented | 1            | 8         | fc_small    |                            |   | 15     | 0.004        |
| ppo2  | random | distances | 1            | 8         | fc_small    |                            |   | 70     | 0.004        |
| ppo2  | random | segmented | 0            | 128       | fc_small    |                            |   | 250    | 0.04         |
| ppo2  | random | distances | 0            | 128       | fc_small    |                            |   | >80    | >0.008       |
| ppo2  | random | segmented | 1            | 128       | fc_small    |                            |   | 55     | 0.003        |
| deepq | random | segmented | 1            |           | cnn_smaller | 0.2                        |   | >450   | >0.12        |
| deeqp | random | segmented | 1            |           | fc_small    | 0.1                        |   | 420    | 0.13         | -->
