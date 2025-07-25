# import gzip
# import json
# from pprint import pprint

# def load_json_gz(file_path):
#     with gzip.open(file_path, 'rt', encoding='utf-8') as f:
#         return json.load(f)

# if __name__ == "__main__":
#     file_path = "/home/ratul/Workstation/ratul/simlingo/database/simlingo_1_data/data/simlingo/validation_1_scenario/routes_validation/random_weather_seed_2_balanced_150/Town13_Rep0_10_route0_01_11_13_24_48/measurements/0000.json.gz"

#     data = load_json_gz(file_path)
#     pprint(data)


import gzip
import json
import numpy as np
import cv2
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def get_camera_to_ego_matrix(pos, rot_deg):
    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = R.from_euler('xyz', rot_deg, degrees=True).as_matrix()
    return T

def load_json_gz(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        return json.load(f)

def get_camera_intrinsics(image_width, image_height, fov_deg):
    fx = fy = image_width / (2.0 * np.tan(fov_deg * np.pi / 360.0)) #(2 * np.tan(np.radians(fov_deg) / 2))
    cx = image_width / 2
    cy = image_height / 2
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ])
    return K

def get_position_from_matrix(matrix):
    return np.array([matrix[0][3], matrix[1][3], matrix[2][3], 1.0])

def project_point(K, point_cam):
    x, y, z = point_cam[:3]
    if x <= 0: return None
    p = K @ np.array([y, -z, x])
    u = p[0] / x
    v = p[1] / x
    return int(u), int(v)

# === Main ===
if __name__ == "__main__":
    base_dir = Path("/home/ratul/Workstation/ratul/simlingo/database/simlingo_1_data/data/simlingo/validation_1_scenario/routes_validation/random_weather_seed_2_balanced_150/Town13_Rep0_10_route0_01_11_13_24_48")
    meas_dir = base_dir / "measurements"
    rgb_dir = base_dir / "rgb"
    out_dir = base_dir / "annotated"
    out_dir.mkdir(exist_ok=True)

    image_w, image_h = 1024, 512
    fov = 110  # adjust if you know exact FOV
    K = get_camera_intrinsics(image_w, image_h, fov)

    # Load all ego matrices
    all_matrices = []
    for i in range(93):
        filename = f"{i:04d}.json.gz"
        data = load_json_gz(meas_dir / filename)
        all_matrices.append(np.array(data["ego_matrix"]))

    camera_to_ego = get_camera_to_ego_matrix([-1.5, 0.0, 2.0], [0.0, 0.0, 0.0])
    ego_to_camera = np.linalg.inv(camera_to_ego)
    
    for i in range(63):  # Only go up to 42 to allow +50 steps
        img_path = rgb_dir / f"{i:04d}.jpg"
        image = cv2.imread(str(img_path))
        
        ego_matrix_curr = all_matrices[i]
        # Full world to camera = ego0_to_camera @ world_to_ego0
        world_to_camera = ego_to_camera @ np.linalg.inv(ego_matrix_curr)

        for j in range(i, min(i+20, len(all_matrices))):
            pos_world = get_position_from_matrix(all_matrices[j])
            pos_cam = world_to_camera @ pos_world
            uv = project_point(K, pos_cam)
            if uv is not None:
                cv2.circle(image, uv, radius=4, color=(0, 255, 0), thickness=-1)

        cv2.imwrite(str(out_dir / f"{i:04d}.jpg"), image)
        print(f"Saved {i:04d}.jpg with future ego positions.")
