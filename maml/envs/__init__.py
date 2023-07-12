from gymnasium.envs.registration import register

register(
    'HalfCheetahVel-v2',
    entry_point='maml.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'maml.envs.mujoco.half_cheetah:HalfCheetahVelEnv'}
)