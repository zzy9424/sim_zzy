import safety_gymnasium
import yaml

file_path="output.yaml"
with open(file_path, 'r') as file:
    wall_config = yaml.safe_load(file)
print(wall_config)
config = {
    'lidar_conf.max_dist': 3,
    'lidar_conf.num_bins': 16,
    'placements_conf.extents': [-1.5, -1.5, 1.5, 1.5],
    'Hazards': {
        'num': 1,
        'size': 0.7,
        'locations': [(0, 0)],
        'is_lidar_observed': True,
        'is_constrained': True,
        'keepout': 0.705,
    },
    'Goal': {
        'keepout': 0.305,
        'size': 0.3,
        'locations': [(1.1, 1.1)],
        'is_lidar_observed': True,
    },
}
config["Walls"]= wall_config[0]["walls"]
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
