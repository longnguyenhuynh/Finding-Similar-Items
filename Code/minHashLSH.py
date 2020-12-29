import argparse
import sys
from os import listdir
from os.path import isfile, join
from typing import Dict, List, Tuple
import random
import numpy as np
from PIL import Image
import pathos.multiprocessing as mp
import time
# import matplotlib.pyplot as plt
# import cv2
pool: mp.Pool


def jaccard(x, y, cpa, cpb):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return (cpa, cpb, intersection_cardinality/float(union_cardinality))


def make_random_hash_fn(tag, p=2**33-355, m=4294967295):
    a = random.randint(1, p-1)
    b = random.randint(0, p-1)
    return lambda x: ((a * x + b) % p) % m


def make_minhash_signature(data, hash_funcs):
    # generate hash functions
    rows, cols, sigrows = len(data), len(data[0]), len(hash_funcs)
    # initialize signature matrix with maxint
    sigmatrix = []
    for i in range(sigrows):
        sigmatrix.append([np.inf] * cols)

    for r in range(rows):
        hashvalue = list(map(lambda x: x(r), hash_funcs))
        # if data != 0 and signature > hash value, replace signature with hash value
        for c in range(cols):
            if data[r][c] == 0:
                continue
            for i in range(sigrows):
                if sigmatrix[i][c] > hashvalue[i]:
                    sigmatrix[i][c] = hashvalue[i]
    return sigmatrix


def calculate_signature(image_file: str, hash_size: int, hash_funcs) -> np.ndarray:
    try:
        pil_image = Image.open(image_file).convert("L").resize(
            (hash_size+1, hash_size),
            Image.ANTIALIAS)
        pix = np.array(pil_image) > 128
        signature = np.array(make_minhash_signature(pix, hash_funcs)).flatten()
        pil_image.close()
        return (image_file, signature)
    except IOError as e:
        raise e


def find_near_duplicates(input_dir: str, threshold: float, hash_size: int, bands: int, hash_funcs) -> List[Tuple[str, str, float]]:
    """
    Find near-duplicate images

    Args:
        input_dir: Directory with images to check
        threshold: Images with a similarity ratio >= threshold will be considered near-duplicates
        hash_size: Hash size to use, signatures will be of length hash_size^2
        bands: The number of bands to use in the locality sensitve hashing process

    Returns:
        A list of near-duplicates found. Near duplicates are encoded as a triple: (filename_A, filename_B, similarity)
    """
    rows: int = int(hash_size**2/bands)
    signatures = dict()
    hash_buckets_list: List[Dict[str, List[str]]] = [dict()
                                                     for _ in range(bands)]

    # Build a list of candidate files in given input_dir
    try:
        file_list = [join(input_dir, f) for f in listdir(
            input_dir) if isfile(join(input_dir, f))]
    except OSError as e:
        raise e

    # Iterate through all files in input directory
    signature = pool.starmap(calculate_signature, [(
        fh, hash_size, hash_funcs) for fh in file_list])

    for sig in signature:
        signatures[sig[0]] = sig[1]
        # Locality Sensitive Hashing
        for i in range(bands):
            signature_band = sig[1][i*rows:(i+1)*rows]
            signature_band_bytes = signature_band.tobytes()
            if signature_band_bytes not in hash_buckets_list[i]:
                hash_buckets_list[i][signature_band_bytes] = list()
            hash_buckets_list[i][signature_band_bytes].append(sig[0])

    # Build candidate pairs based on bucket membership
    candidate_pairs = set()
    for hash_buckets in hash_buckets_list:
        for hash_bucket in hash_buckets.values():
            if len(hash_bucket) > 1:
                hash_bucket = sorted(hash_bucket)
                for i in range(len(hash_bucket)):
                    for j in range(i+1, len(hash_bucket)):
                        candidate_pairs.add(
                            tuple([hash_bucket[i], hash_bucket[j]])
                        )
    # Check candidate pairs for similarity (Parallel)
    similarity = pool.starmap(
        jaccard, [(signatures[cpa], signatures[cpb], cpa, cpb) for cpa, cpb in candidate_pairs])
    near_duplicates = [(similar[0], similar[1], similar[2])
                       for similar in similarity if similar[2] > threshold]

    # Sort near-duplicates by descending similarity and return
    near_duplicates.sort(key=lambda x: x[2], reverse=True)
    return near_duplicates


def main(argv):
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Efficient detection of near-duplicate images using locality sensitive hashing")
    parser.add_argument("-i", "--inputdir", type=str, default="",
                        help="directory containing images to check")
    parser.add_argument("-t", "--threshold", type=float,
                        default=0.8, help="similarity threshold")
    parser.add_argument("-s", "--hash-size", type=int, default=32,
                        help="hash size to use, signature length = hash_size^2", dest="hash_size")
    parser.add_argument("-b", "--bands", type=int,
                        default=32, help="number of bands")

    args = parser.parse_args()
    input_dir = args.inputdir
    threshold = args.threshold
    hash_size = args.hash_size
    bands = args.bands

    hash_funcs = pool.map(make_random_hash_fn, range(hash_size**2))

    try:
        near_duplicates = find_near_duplicates(
            input_dir, threshold, hash_size, bands, hash_funcs)
        if near_duplicates:
            print(
                f"Found {len(near_duplicates)} near-duplicate images in {input_dir} (threshold {threshold:.2%})")
            for a, b, s in near_duplicates:
                print(f"{s:.2%} similarity: file 1: {a} - file 2: {b}")

                # Uncomment to see picture
                # plt.subplot(121), plt.imshow(cv2.imread(a))
                # plt.title('Similar'), plt.xticks([]), plt.yticks([])

                # plt.subplot(122), plt.imshow(cv2.imread(b))
                # plt.title('Original'), plt.xticks([]), plt.yticks([])
                # plt.show()
        else:
            print(
                f"No near-duplicates found in {input_dir} (threshold {threshold:.2%})")
    except OSError:
        print(f"Couldn't open input directory {input_dir}")


if __name__ == "__main__":
    start = time.time()
    pool = mp.Pool(processes=mp.cpu_count())
    main(sys.argv)
    pool.close()
    end = time.time()
    print(end - start)
