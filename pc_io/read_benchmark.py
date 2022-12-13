import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


import argparse
import time

from utils import read_json

def main(fpath, use_custom, limit):
    data = read_json(fpath)
    obj_paths = []
    for idx, scene_dict in data["scenes"].items():
        obj_paths.append(scene_dict["point_cloud"])

    if limit:
        obj_paths = obj_paths[:limit]

    if use_custom:
        from pc_io.read_save import load_obj
    else:
        from pytorch3d.io import load_obj

    start = time.time()
    for path in obj_paths:
        load_obj(path)
    finish = time.time()
    total_time = finish - start
    avg_time = total_time / len(obj_paths)


    print("Total Time: {:.4f}".format(total_time))
    print("Average Time: {:4f}".format(avg_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trn-path", type=str, help="path to train metadata")
    parser.add_argument("--custom", action="store_true", help="path to train metadata")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    main(
        args.trn_path,
        args.custom,
        args.limit
    )