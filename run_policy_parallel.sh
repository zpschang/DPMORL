mkdir -p outputs
for env in $(cat env.txt); do
    nohup python -u main_policy.py --lamda=0.1 --env $env \
        --reward_two_dim --exp_name dpmorl > outputs/${env}.txt 2>&1 &
    echo "Running $env"
    sleep 4s
done