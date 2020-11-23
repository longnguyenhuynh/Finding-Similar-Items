import sys
from PIL import Image
import cv2
import numpy as np  
import os 
import pathlib
from pathlib import Path
import imagehash

def calculate_signature(image_file: str, hash_size: int) -> np.ndarray:
    """ 
    Calculate the dhash signature of a given file
    
    Args:
        image_file: the image (path as string) to calculate the signature for
        hash_size: hash size to use, signatures will be of length hash_size^2
    
    Returns:
        Image signature as Numpy n-dimensional array or None if the file is not a PIL recognized image
    """
    try:
        pil_image = Image.open(image_file).convert("L").resize(
                            (hash_size+1, hash_size), 
                            Image.ANTIALIAS)
        dhash = imagehash.dhash(pil_image, hash_size)
        signature = dhash.hash.flatten()
        pil_image.close()
        return signature.astype(int) 
    except IOError as e:
        raise e

def jaccard(arr1: np.ndarray, arr2: np.ndarray) -> float:
    count = 0
    for i in range(len(arr1)):
        if arr1[i] == arr2[i]:
            count += 1
    return count/len(arr1)

def minhash(image_file: str, threshold=0.7) -> dict:
    img_dict = {}
    signature = calculate_signature(image_file, 5)
    root_path = Path(__file__).parent.absolute()
    path = os.path.join(root_path, 'cup')
    for item in os.listdir(path):
        img_sig = calculate_signature('cup/'+item, 5)
        print(img_sig)
        jaccard_value = jaccard(signature, img_sig)
        if jaccard_value >= threshold:
            img_dict.update({item: jaccard_value})
    return img_dict

if __name__ == "__main__":
    img_similar_dict = minhash('cup/image_0013.jpg')
    for item in img_similar_dict.items():
        print(item)