import safety_gymnasium
import yaml

file_path="output.yaml"
with open(file_path, 'r') as file:
    config = yaml.safe_load(file)

env_id = 'SafetyPointGoalBase-v0'
env = safety_gymnasium.make(env_id, render_mode="human",config=config)
# env = safety_gymnasium.make(env_id)
obs, info = env.reset()
while True:
    act = env.action_space.sample()
    obs, reward, cost, terminated, truncated, info = env.step(act)
    if terminated or truncated:
        break
    # env.render()
