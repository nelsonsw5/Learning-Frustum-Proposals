import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


import argparse
import time
import torch


from pc_io.read_save import load_obj

TMP_FILE = "./tmp.obj"

def main(fpath, use_custom, n):
    pc, _ = load_obj(fpath)

    start = time.time()
    for i in range(n):
        if use_custom:
            from pc_io.read_save import write_obj
            write_obj(pc, TMP_FILE)
        else:
            from pytorch3d.io import save_obj
            save_obj(TMP_FILE, pc, faces=torch.LongTensor())

    finish = time.time()
    total_time = finish - start
    avg_time = total_time / n


    print("Total Time: {:.4f}".format(total_time))
    print("Average Time: {:4f}".format(avg_time))

    os.remove(TMP_FILE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pc-path", type=str, help="path to train metadata")
    parser.add_argument("--custom", action="store_true", help="path to train metadata")
    parser.add_argument("--n", type=int, help="Number of times to write")
    args = parser.parse_args()

    main(
        args.pc_path,
        args.custom,
        args.n
    )