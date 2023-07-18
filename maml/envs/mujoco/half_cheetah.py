import numpy as np
from gymnasium.envs.mujoco import HalfCheetahEnv as HalfCheetahEnv_

class HalfCheetahEnv(HalfCheetahEnv_):
    def viewer_setup(self):
        # 调整视角的，也可以不调整
        camera_id = self.model.camera_name2id('track')
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        self.viewer._hide_overlay = True
    
    def render(self, mode='human'):
        # 不用管
        if mode == 'rgb_array':
            self._get_viewer().render()
            width, height = 500, 500
            data = self._get_viewer().read_pixels(width, height, depth=False)
            return data
        elif mode == 'human':
            self._get_viewer(mode='human').render()

class HalfCheetahVelEnv(HalfCheetahEnv):
    def __init__(self, task={}, low=0.0, high=2.0):
        self._task = task
        self.low = low
        self.high = high
        self._goal_vel = task.get('velocity', 0.0)
        super(HalfCheetahVelEnv, self).__init__()

    def step(self, action):
        # 改写step, 是为了让每次学习的任务都不同, 这里的任务是指目标速度
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum() # 惩罚力的平方
        reward_run = - np.abs(xposafter - self._goal_vel) / self.dt # 奖励与目标速度的差的绝对值
        reward = reward_ctrl + reward_run
        infos = dict(reward_run=reward_run, reward_ctrl=reward_ctrl, task=self._task)
        return ob, reward, False, False, infos

    def sample_tasks(self, num_tasks):
        # 随机采样num_tasks个任务，每个任务都是一个字典，字典中只有一个键值对，键是'velocity'，值是一个随机数
        velocities = self.np_random.uniform(self.low, self.high, size=(num_tasks,)) 
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks
    
    def reset_task(self, task):
        self._task = task
        self._goal_vel = task['velocity']