import setup_path
import gym
import thyroid_glanod_gym
import time
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# import os
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "thyroid_glanod_gym:thyroid_glanod-discrete-cjp-v0",
                image_shape=(64, 64, 1),
            )
        )
    ]
)

filename = "thyroid_glanod_dqn_cjp_new2"

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
model = DQN(
    "CnnPolicy",
    env,
    buffer_size = 100000,
    learning_starts  = 10000,
    target_update_interval = 2500,
    exploration_fraction = 0.3,
    verbose=1,
    device="cuda",
    tensorboard_log=f"./logs/{filename}_logs/",
)

# model.load("./best_model.zip")

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=10,
    best_model_save_path=f"./model/{filename}_model",
    log_path=".",
    eval_freq=2500,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=1e5,
    progress_bar=True,
    **kwargs
)

# Save policy weights
model.save(f"./saved_policy/{filename}_policy")

# model.load("/model/dqn_goal_plus_static_model/best_model.zip")
# model.predict()