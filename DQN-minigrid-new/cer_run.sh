#
#i=1000
#change_timestep=50000
#max_timestep=300000
#replay_max_size=50000
#epsilon_decay_step=100000
#batch_size=64
#noise_amp=1
#for seed in 668 669 670
#do
#    python train.py --env_name "MiniGrid-SimpleCrossingS9N1-v0" \
#    --note noisy_obs_CER\
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
#    --noise_amp $noise_amp \
#    --change_type noisy_obs
#    i=$((i+1))
#done

i=1003
change_timestep=50000
max_timestep=300000
replay_max_size=50000
epsilon_decay_step=200000
batch_size=64
noise_amp=0.75
for seed in 111 222 333
do
    python train.py --env_name "MiniGrid-LavaCrossingS9N1-v0" \
    --note noisy_obs_CER\
    --number ${i} \
    --random_seed $seed \
    --model_size large \
    --epsilon_init 1 \
    --epsilon_decay_step ${epsilon_decay_step} \
    --replay_max_size ${replay_max_size} \
    --batch_size $batch_size  \
    --max_timestep  ${max_timestep}  \
    --replay CER \
    --full_obs 1 \
    --lambda_init 1 \
    --change_env 1 \
    --change_timestep ${change_timestep} \
    --noise_amp $noise_amp \
    --change_type noisy_obs
    i=$((i+1))

    python train.py --env_name "MiniGrid-LavaCrossingS9N1-v0" \
    --note noisy_obs_PER_0.5\
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
    --alpha 0.7 \
    --beta 0.7 \
    --lambda_init 1 \
    --change_env 1 \
    --change_timestep  ${change_timestep}  \
        --noise_amp $noise_amp\
    --change_type noisy_obs
      i=$((i+1))
done


#
#i=1000
#change_timestep=50000
#max_timestep=300000
#replay_max_size=50000
#epsilon_decay_step=150000
#batch_size=128
#for seed in  4 5 6
#do
#for noise_amp in 1.2
#do
#    python train.py --env_name "MiniGrid-Door-6x6-v0" \
#    --note noisy_obs_CER_noise_${noise_amp}\
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
#    --noise_amp $noise_amp \
#    --change_type noisy_obs
#    i=$((i+1))
#    done
#done