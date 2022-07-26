python fed_mda_simplified.py --num_users_per_domain 5 --bs 32 --local_bs 32 --lr 0.01 --epochs 20 --local_ep 2 --timesteps 20 --eval_every 1 --iid --gpu 0 --mda_threshold_frac 1 --wandb ann_5users_1tdf_iid
#########################
python fed_mda_simplified.py --num_users_per_domain 5 --bs 32 --local_bs 32 --lr 0.01 --epochs 20 --local_ep 2 --timesteps 20 --eval_every 1 --iid --gpu 0 --mda_threshold_frac 0.25 --wandb ann_5users_0.25tdf_iid
#########################