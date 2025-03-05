from gymnasium.envs.registration import register

register(
    id="so_arm_rl/SoFetchEnv-v0",
    entry_point="so_arm_rl.envs.fetch:SoFetchEnv",
)