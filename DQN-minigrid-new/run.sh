i=1
change_timestep=50000
max_timestep=1000000
replay_max_size=50000
epsilon_decay_step=200000
batch_size=64
noise_amp=0.5
seed=1
python train.py --env_name "MiniGrid-Door-5x5-v0" \
    --note noisy_obs_default\
    --number ${i} \
    --random_seed $seed \
    --model_size large \
    --epsilon_init 1 \
    --epsilon_decay_step ${epsilon_decay_step} \
    --replay_max_size ${replay_max_size} \
    --batch_size $batch_size  \
    --max_timestep  ${max_timestep}  \
    --replay default \
    --full_obs 1 \
    --lambda_init 1 \
    --change_env -1
    i=$((i+1))