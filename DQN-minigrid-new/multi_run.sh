i=6
change_timestep=50000
max_timestep=300000
replay_max_size=50000
epsilon_decay_step=150000
batch_size=128
for seed in 667 668
do
#    python train.py --env_name "MiniGrid-Door-6x6-v0" \
#    --note obstacle_default\
#    --number ${i} \
#    --random_seed $seed \
#    --model_size large \
#    --epsilon_init 1 \
#    --epsilon_decay_step ${epsilon_decay_step} \
#    --replay_max_size ${replay_max_size} \
#    --batch_size $batch_size  \
#    --max_timestep  ${max_timestep}  \
#    --replay default \
#    --full_obs 1 \
#    --lambda_init 1 \
#    --change_env 1 \
#    --change_timestep ${change_timestep} \
#    --change_type obstacle
#    i=$((i+1))
  for alpha in 0.7
  do
      python train.py --env_name "MiniGrid-Door-6x6-v0"\
    --note obstacle_EPER\
    --number ${i} \
    --random_seed $seed \
    --model_size large \
    --epsilon_init 1 \
    --epsilon_decay_step ${epsilon_decay_step} \
    --replay_max_size ${replay_max_size} \
    --batch_size $batch_size \
    --max_timestep  ${max_timestep}  \
    --replay EPER \
    --full_obs 1 \
    --alpha ${alpha} \
    --beta 1 \
    --beta_min 0.01 \
    --beta_decay_step 100000 \
    --lambda_init 0.8 \
    --change_env 1 \
    --change_timestep ${change_timestep}  \
    --change_type obstacle
    i=$((i+1))

    python train.py --env_name "MiniGrid-Door-6x6-v0"\
    --note obstacle_PER\
    --number ${i} \
    --random_seed $seed \
    --model_size large \
    --epsilon_init 1 \
    --epsilon_decay_step ${epsilon_decay_step} \
    --replay_max_size ${replay_max_size} \
    --batch_size $batch_size \
    --max_timestep  ${max_timestep}  \
    --replay PER \
    --full_obs 1 \
    --alpha ${alpha} \
    --beta 1 \
    --beta_min 0.01 \
    --beta_decay_step 100000 \
    --lambda_init 0.8 \
    --change_env 1 \
    --change_timestep ${change_timestep}  \
    --change_type obstacle
    i=$((i+1))
done
done
#
#  python train.py --env_name "MiniGrid-Door-6x6-v0" \
#    --note obstacle_CER\
#    --number ${i} \
#    --random_seed $seed \
#    --model_size large \
#    --epsilon_init 1 \
#    --epsilon_decay_step ${epsilon_decay_step} \
#    --replay_max_size ${replay_max_size} \
#    --batch_size $batch_size  \
#    --max_timestep  ${max_timestep}  \
#    --replay CER \
#    --full_obs 1 \
#    --lambda_init 1 \
#    --change_env 1 \
#    --change_timestep ${change_timestep} \
#    --change_type obstacle
#    i=$((i+1))
#done