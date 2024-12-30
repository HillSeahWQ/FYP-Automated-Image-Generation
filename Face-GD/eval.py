from pathlib import Path
import os
from tqdm import tqdm
from functions.clip.base_clip import CLIPEncoder
from functions.face_parsing.model import FaceParseTool
from functions.landmark.model import FaceLandMarkTool
from functions.arcface.model import IDLoss
from functions.anime2sketch.model import FaceSketchTool

def get_image_paths(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


def evaluate(generated_image_directory, conditions):

    distances = {key: {"l2": [], "fid":[]} for key in list(conditions.keys())}
    img_paths = get_image_paths(generated_image_directory)

    if "clip" in list(conditions.keys()):
        clip_encoder = CLIPEncoder().cuda()
    if "parse" in list(conditions.keys()):
        parser = FaceParseTool(ref_path=conditions['parse']).cuda()
    if "sketch" in list(conditions.keys()):
        img2sketch = FaceSketchTool(ref_path=conditions['sketch']).cuda()
    if "landmark" in list(conditions.keys()):
        img2landmark = FaceLandMarkTool(ref_path=conditions['landmark']).cuda()
    if "arc" in list(conditions.keys()):
        idloss = IDLoss(ref_path=conditions['arc']).cuda()

    for img_path in tqdm(img_paths):

        if "clip" in list(conditions.keys()):
            distances["clip"]["l2"].append(clip_encoder.calculate_euclidean_distance(image_path=img_path, text=conditions['clip']))
            distances["clip"]["fid"].append(clip_encoder.calculate_fid(image_path=img_path, text=conditions['clip']))

        if "parse" in list(conditions.keys()):
            distances["parse"]["l2"].append(parser.calculate_mask_distance(conditions['parse'], img_path))
            distances["parse"]["fid"].append(parser.calculate_fid(conditions['parse'], img_path))

        if "sketch" in list(conditions.keys()):
            distances["sketch"]["l2"].append(img2sketch.calculate_sketch_distance(conditions['sketch'], img_path))
            distances["sketch"]["fid"].append(img2sketch.calculate_fid(conditions['sketch'], img_path))

        if "landmark" in list(conditions.keys()):
            distances["landmark"]["l2"].append(img2landmark.calculate_landmark_distance(conditions['landmark'], img_path))
            distances["landmark"]["fid"].append(img2landmark.calculate_fid(conditions['landmark'], img_path))

        if "arc" in list(conditions.keys()):
            distances["arc"]["l2"].append(idloss.calculate_id_distance(conditions['arc'], img_path))
            distances["arc"]["fid"].append(idloss.calculate_fid(conditions['arc'], img_path))


    for key, val in distances.items():
        print(f"{key} -> mean L2 distance = {sum(val['l2'])/len(img_paths)}")
        print(f"{key} -> mean FID distance = {sum(val['fid'])/len(img_paths)}")


# To edit
generated_image_directory = Path.cwd() / "exp" / "res" / "clip_landmark_ref2334jpg" / "no_int_terms"
conditions = {
    "clip": "black woman",
    # "parse": "./images/294.jpg",
    "landmark": "./images/2334.jpg",
    # "arc": "./images/id10.png",
    # "sketch": "./images/294.jpg"
}

if __name__ == "__main__":
    evaluate(generated_image_directory, conditions)