import os
from pathlib import Path
from functions.face_parsing.model import FaceParseTool


def handle_image_paths(img_ref_path, img_output_folder):
    if not os.path.exists(img_ref_path):
        raise FileNotFoundError(f"Error {img_ref_path} path not found")
    os.makedirs(img_output_folder, exist_ok=True)


def save_segmentation_map(img_ref_path, img_output_folder):
    handle_image_paths(img_ref_path, img_output_folder)
    face_parse_tool = FaceParseTool(ref_path=img_ref_path)
    face_parse_tool.save_segmentation_map(img_ref_path, img_output_folder/ "segmap.jpg")


def main(
        img_ref_path = Path().cwd() / "images" / "294.jpg",
        img_output_folder = Path().cwd() / "exp" / "conditions"
):
    save_segmentation_map(img_ref_path, img_output_folder)


if __name__ == "__main__":
    main()