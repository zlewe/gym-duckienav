from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='DuckieNav-v0',
    entry_point='gym_duckienav.envs:DuckieNavEnv',
)

