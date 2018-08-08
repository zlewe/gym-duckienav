from gym.envs.registration import register

register(
    id='DuckieNav-v0',
    entry_point='gym_duckienav.envs:DuckieNavEnv',
)

register(
    id='DuckieNav-v2',
    entry_point='gym_duckienav.envs:DuckieNavEnvV2'
)

