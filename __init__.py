from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='DuckieNavTaxi-v0',
    entry_point='gym_duckienav_taxi.envs:DuckieNavTaxiEnv',
)

