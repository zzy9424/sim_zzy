

import safety_gymnasium

env_id = 'SafetyCarGoal0-v0'
env = safety_gymnasium.make(env_id, render_mode="human")
obs, info = env.reset()
while True:
    act = env.action_space.sample()
    obs, reward, cost, terminated, truncated, info = env.step(act)
    if terminated or truncated:
        break
    # env.render()