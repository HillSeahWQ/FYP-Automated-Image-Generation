import os
from pathlib import Path
from functions.face_parsing.model import FaceParseTool
from functions.anime2sketch.model import FaceSketchTool
from functions.landmark.model import FaceLandMarkTool


def handle_image_paths(img_ref_path, img_output_folder):
    if not os.path.exists(img_ref_path):
        raise FileNotFoundError(f"Error {img_ref_path} path not found")
    os.makedirs(img_output_folder, exist_ok=True)


def save_segmentation_map(img_ref_path, img_output_folder):
    handle_image_paths(img_ref_path, img_output_folder)
    face_parse_tool = FaceParseTool(ref_path=img_ref_path)
    face_parse_tool.save_segmentation_map(img_ref_path, img_output_folder/ "segmap.jpg")

def save_sketch(img_ref_path, img_output_folder):
    handle_image_paths(img_ref_path, img_output_folder)
    face_sketch_tool = FaceSketchTool(ref_path=img_ref_path)
    face_sketch_tool.save_sketch(img_ref_path, img_output_folder/ "sketch.jpg")

def save_landmarks(img_ref_path, img_output_folder):
    handle_image_paths(img_ref_path, img_output_folder)
    face_landmark_tool = FaceLandMarkTool(ref_path=img_ref_path)
    face_landmark_tool.save_landmarks(img_output_folder/ "landmarks.jpg")


def main(
        img_ref_folder = Path().cwd() / "images",
        img_output_folder = Path().cwd() / "exp" / "conditions"
):
    save_segmentation_map(
        img_ref_path=img_ref_folder/"bm.",
        img_output_folder=img_output_folder
    )
    # save_sketch(
    #     img_ref_path=img_ref_folder/"image.png", 
    #     img_output_folder=img_output_folder
    # )
    # save_landmarks(
    #     img_ref_path="./images/2334.jpg", 
    #     img_output_folder=img_output_folder
    # )


if __name__ == "__main__":
    main()
