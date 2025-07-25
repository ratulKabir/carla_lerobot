import gzip
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def load_json_gz(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        return json.load(f)

def extract_state(matrix, speed):
    """Extracts position + RPY + speed from ego matrix"""
    pos = [matrix[0][3], matrix[1][3], matrix[2][3]]
    rot = np.array(matrix)[:3, :3]
    rpy = R.from_matrix(rot).as_euler('xyz', degrees=False).tolist()
    return pos + rpy + [speed]

def build_lerobot_table(meas_dir, future_steps=10, fps=10, task_index=0, episode_index=0):
    meas_dir = Path(meas_dir)
    json_files = sorted(meas_dir.glob("*.json.gz"))
    n_frames = len(json_files)

    all_matrices = []
    all_speeds = []

    for f in json_files:
        data = load_json_gz(f)
        all_matrices.append(np.array(data["ego_matrix"]))
        all_speeds.append(data["speed"])

    action_list = []
    action_is_pad_list = []
    obs_state_list = []
    obs_state_is_pad_list = []
    timestamp_list = []
    frame_index_list = []
    episode_index_list = []
    index_list = []
    task_index_list = []
    top_is_pad_list = []
    task_list = []

    for i in tqdm(range(n_frames)):
        obs_matrix = all_matrices[i]
        obs_speed = all_speeds[i]
        obs_state = extract_state(obs_matrix, obs_speed)
        obs_state_list.append(obs_state)
        obs_state_is_pad_list.append([False])

        T_i_inv = np.linalg.inv(obs_matrix)
        future = []
        is_pad = []

        for j in range(i + 1, i + 1 + future_steps):
            if j >= n_frames:
                future.append([0.0]*7)
                is_pad.append(True)
            else:
                T_j = all_matrices[j]
                T_rel = T_i_inv @ T_j
                pos_rel = T_rel[:3, 3].tolist()
                rpy_rel = R.from_matrix(T_rel[:3, :3]).as_euler('xyz', degrees=False).tolist()
                speed_j = all_speeds[j]
                future.append(pos_rel + rpy_rel + [speed_j])
                is_pad.append(False)

        action_list.append(future)
        action_is_pad_list.append(is_pad)
        timestamp_list.append(i / fps)
        frame_index_list.append(i)
        episode_index_list.append(episode_index)
        index_list.append(i)
        task_index_list.append(task_index)
        top_is_pad_list.append(False)
        task_list.append("")

    # Build Arrow table
    table = pa.table({
        "action": pa.array(action_list, type=pa.list_(pa.list_(pa.float32(), 7))),
        "action_is_pad": pa.array(action_is_pad_list, type=pa.list_(pa.bool_())),
        "observation.state": pa.array(obs_state_list, type=pa.list_(pa.float32(), 7)),
        "observation.state_is_pad": pa.array(obs_state_is_pad_list, type=pa.list_(pa.bool_())),
        "timestamp": pa.array(timestamp_list, type=pa.float32()),
        "frame_index": frame_index_list,
        "episode_index": episode_index_list,
        "index": index_list,
        "task_index": task_index_list,
        "observation.images.top_is_pad": top_is_pad_list,
        "task": task_list,
    })

    return table

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--meas_dir", default="/home/ratul/Workstation/ratul/simlingo/database/simlingo_1_data/data/simlingo/validation_1_scenario/routes_validation/random_weather_seed_2_balanced_150/Town13_Rep0_10_route0_01_11_13_24_48/measurements")
    parser.add_argument("--save_path", default="/home/ratul/Workstation/ratul/database/lerobot/carla_lerobot/data/chunk-000/episode_000000.parquet")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--future_steps", type=int, default=30)
    args = parser.parse_args()

    table = build_lerobot_table(
        meas_dir=args.meas_dir,
        future_steps=args.future_steps,
        fps=args.fps,
        task_index=0,
        episode_index=0
    )

    pq.write_table(table, args.save_path)
    print(f"✅ Saved {table.num_rows} samples to {args.save_path}")
