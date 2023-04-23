# we will use triple_loss for the model,
# we need a triplet generator to generate the triplets for the model.
# so, for each image in mtccn-faces (anchor), we will chooese
# N random images from the same class (positive), and N random images from different classes (negative)


import os
import random
import numpy as np
from pyprojroot.here import here

def generate_triplets(path, num_triplets_each_person=100):
    """
    Generate triplets for training a triplet-based model using anchor, positive, and negative images.

    Parameters:
        path (str): Path to the directory containing the anchor, positive, and negative images.
        num_triplets_each_person (int, optional, default=100): Number of triplets to generate for each person.

    Returns:
        tuple: A tuple containing three lists - anchors, positives, and negatives.
               Each list contains file paths to the anchor, positive, and negative images, respectively.

    Dependencies:
        - os: For working with file paths.
        - random: For generating random numbers.
        - numpy (imported as np): For working with arrays.
        - pyprojroot.here: For getting the root directory path of the project.

    Usage:
        anchors, positives, negatives = generate_triplets(path, num_triplets_each_person=100)

    """

    anchors = []
    positives = []
    negatives = []

    for folder in os.listdir(path):
        for i in range(num_triplets_each_person):
            anchor = random.choice(os.listdir(os.path.join(path, folder)))
            positive = random.choice(os.listdir(os.path.join(path, folder)))
            other_folder = random.choice([f for f in os.listdir(path) if f != folder])
            negative = random.choice(os.listdir(os.path.join(path,other_folder )))
            anchors.append(os.path.join(path, folder, anchor))
            positives.append(os.path.join(path, folder, positive))
            negatives.append(os.path.join(path, other_folder, negative))

    return (anchors, positives, negatives)

if __name__ == '__main__':
    anchros, positives, negatives = generate_triplets(here("mtcnn-faces"), num_triplets_each_person=100)
    print(anchros[0], positives[0], negatives[0])
