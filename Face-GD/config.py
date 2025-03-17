# Static dictionary for arguments
ARGUMENTS = {
    "seed": 1234,
    "exp": "exp",
    "doc": "celeba_hq",
    "comment": "",
    "verbose": "info",
    "sample": False,
    "image_folder": "multi_cond",
    "ni": True,
    "timesteps": 100,
    "model_type": "face",
    "batch_size": 100,
    "class_num": 10,
    "sample_strategy": "clip_ddim",
    "mu": 1.0,
    "rho_scale": 0.2, #  text - 0.2 (MAIN), parse - 0.2, sketch - 20, landmark- 500, id - 100
    "prompt": "black woman",
    "stop": 200,
    "ref_path": "./images/294.jpg",
    "ref_path2": None,
    "scale_weight": None,
    "rt": 1
}

conditions = {
    "clip": "black woman",
    "parse": "./images/294.jpg", # 294.jpg
    # "arc": "./images/id10.png", #id10.png
    # "landmark": "./images/2334.jpg" # 2334.jpg
    # "sketch": "./images/294.jpg" # 294.jpg
}