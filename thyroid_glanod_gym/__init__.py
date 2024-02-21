from gym.envs.registration import register

register(
    id="thyroid_glanod-discrete-cjp-v0", entry_point="thyroid_glanod_gym.envs:ThyroidGlaNodDiscreteCJPEnv",
)
