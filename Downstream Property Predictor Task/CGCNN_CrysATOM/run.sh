python train.py --epochs 400 --data-path 'jarvis/formation_energy_peratom' --lr 0.01 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --checkpoint_path './formation_energy_peratom' --atomvector_path "./dense_vector/CrysAtom_vector.pkl"