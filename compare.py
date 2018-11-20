#!/usr/bin/env python

from feature_extraction import extract_prelogits_fv

from scipy.spatial.distance import euclidean
import magic
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
            '--image', type=str,  required=True,
                help='an image to compare'
                )
parser.add_argument(
            '--other_images_dir', type=str, required=True,
                help='a directory of other images to compare'
                )


if __name__ == "__main__":
    args = parser.parse_args()
    first = args.image
    other_dir = args.other_images_dir
   
    # extract prelogits from one file
    first_prelogits = extract_prelogits_fv(first)
   
    other_images = []
    for file in sorted(os.listdir(other_dir)):
        full_path = os.path.join(other_dir, file)
        mime_type = magic.from_file(full_path)
        if "image" in mime_type:
            other_images.append(full_path)
        else:
            print("skipping {} - not an image".format(full_path))
    
    print(other_images)
    other_prelogits = [extract_prelogits_fv(img) for img in other_images]
    distances = [euclidean(first_prelogits, other_prelogits[i]) for i in range(len(other_prelogits))]
    
    nearest_idx = np.argmin(distances)
    print("nearest image is {}, distance: {}".format(
        other_images[nearest_idx],
        distances[nearest_idx]
    ))

    furthest_idx = np.argmax(distances)
    print("furthest image is {}, distance: {}".format(
        other_images[furthest_idx],
        distances[furthest_idx]
    ))
