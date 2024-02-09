from matformer.train_props import train_prop_model 
props = [
    "e_form",
    "gap pbe",
]
train_prop_model(learning_rate=0.001,name="matformer", dataset="megnet", prop=props[0], pyg_input=True, n_epochs=500, batch_size=64, use_lattice=True, output_dir="./demo", use_angle=False, save_dataloader=False, atom_features = "./dense_vector/200_dimensional_vector.pkl",random_seed=123,
                )