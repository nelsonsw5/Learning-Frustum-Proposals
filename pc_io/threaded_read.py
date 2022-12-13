import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
from threading import Thread
from typing import Callable, List
import numpy as np
import numba
import torch

from pytorch3d.io import load_obj
from read_save.pc_io import load_obj
from utils import read_json
import pandas as pd
import time



def my_load_obj(fpath):
    f = open(fpath, "r")
    #os.set_blocking(f.fileno(), False)
    rows = pd.read_csv(fpath, sep=" ", engine="c", header=None)
    verts = rows[rows[0] == "v"]
    verts = torch.tensor(rows.iloc[:, 1:].astype(np.float32).values)
    #print(rows.head())
    #rows = f.readlines()
    #f.close()
    #_parse_obj(rows)
    """n_verts = 0
    for line in rows:
        if line[0] == "v":
            n_verts += 1

    file = np.zeros((n_verts, 3), dtype=object)
    i = 0
    for line in rows:
        row = line.split()
        if row[0] == "v":
            file[i, :] = row[1:]
            i += 1"""

@numba.jit(nopython=True)
def _parse_obj(rows):
    rows = np.char.split(rows)
    n = len(rows)
    n_verts = 0
    for i in range(n):
        row = rows[i]
        if row[0] == "v":
            n_verts += 1

    file = np.zeros((n_verts, 3), dtype=str)
    cntr = 0
    for i in range(n):
        row = rows[i]
        if row[0] == "v":
            file[cntr, :] = row[1:]
            cntr += 1




def run_with_threads(
    obj_paths: List,
    slice_fn: Callable,
    arg_name: str,
    num_threads=25,
    max_obj=1000,
):
    threads = []
    if max_obj:
        obj_paths = obj_paths[:max_obj]
    obj_per_thread = int(len(obj_paths) / num_threads)
    for i in range(num_threads):
        slice = obj_paths[
            i * obj_per_thread : (i + 1) * obj_per_thread
                ]
        thread = Thread(
            target=slice_fn,
            kwargs={arg_name: slice},
        )
        #threads.append(thread)
        thread.start()

def do_stuff(path_list):
    #print(path_list)
    start = time.time()
    for path in path_list:
        #v, _, _ = load_obj(path)
        v, _ = load_obj(path)
        #v = np.loadtxt(path, delimiter=" ", dtype=object)"""
    end = time.time()
    print(end - start)
    #time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data = read_json("/home/porter/data/the-blue-pill/dataset-5k-full-points/test-metadata.json")
    obj_paths = []
    for idx, scene_dict in data["scenes"].items():
        obj_paths.append(scene_dict["point_cloud"])


    if args.thread:
        print("run with threads: True")
        run_with_threads(
            obj_paths[:args.limit] if args.limit else obj_paths,
            do_stuff,
            arg_name="path_list"
        )
    else:
        do_stuff(obj_paths[:args.limit] if args.limit else obj_paths)

