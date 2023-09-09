# Face Recognition with Triplet Loss in TensorFlow

## Overview

This repository contains code for fine-tuning a face recognition model using the Inception ResNet V2 architecture with a triplet loss in TensorFlow. The model is trained to recognize faces of individuals from a dataset, and it uses a face-only dataset obtained using the MTCNN (Multi-task Cascaded Convolutional Networks) face detection system.


## How to Reproduce the Results

To reproduce the results, follow the steps below:

1. Clone the repository:

```
git clone https://github.com/carlesoctav/face-recognition-with-triplet-loss-tensorflow.git
```

2. Install the dependencies: (use any environment management tool of your choice)

```
pip install -r requirements.txt
```

3. Prepare the dataset:

For each person whose face you want the system to recognize, create a folder with the name of the person in the `data` folder, and put all the images of the person in that folder. For example, if you want the system to recognize the face of "Jack", create a folder named "Jack" in the `data` folder, and put all the images of Jack in that folder.

4. Get the face-only dataset using MTCNN:

Run the following command to extract faces from the images in the dataset using MTCNN:

```
python script/face_detection.py
```

5. Train the model:

Train the model to recognize the faces of the individuals in the dataset. The pretrained Inception ResNet V2 model will be fine-tuned only on the classification blocks, using the face-only dataset obtained in the previous step. Use the following command:

```
python script/training_model.py
```

You can check all available arguments using:

```
python script/training_model.py --help
```

6. Create embeddings for each person in the dataset:

Run the following command to create embeddings (feature vectors) for each person in the dataset using the trained model:

```
python script/embedding-space-for-inference.py
```

7. Inference time:

Run the following command to perform inference on new images using the trained model and embedding vectors that were created in the previous step:

```
python script/inference.py
```

## References
[FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)

[https://github.com/pooya-mohammadi/Face ](https://github.com/pooya-mohammadi/Face)

[Image similarity estimation using a Siamese Network with a triplet loss](https://keras.io/examples/vision/siamese_network/)

