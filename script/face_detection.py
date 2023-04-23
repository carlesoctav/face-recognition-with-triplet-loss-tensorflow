import mtcnn
from pyprojroot.here import here
import os
import matplotlib.pyplot as plt
import cv2


# use mtcnn to detect faces in the data folder, make a folder name mtcnn-faces as output, change the output size to size

def mtcnn_detect(path, output_size=160):

    if not os.path.exists(here("mtcnn-faces")):
        os.mkdir(here("mtcnn-faces"))

    for folder in os.listdir(path):

        if not os.path.exists(here(f"mtcnn-faces/{folder}")):
            os.mkdir(here(f"mtcnn-faces/{folder}"))

        output_folder = here(f"mtcnn-faces/{folder}")
        print(output_folder)
        
        for file in os.listdir(os.path.join(path, folder)):
            image = plt.imread(os.path.join(path, folder, file))
            detector = mtcnn.MTCNN()
            result = detector.detect_faces(image) 
            x1, y1, width, height = result[0]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = image[y1:y2, x1:x2]
            image_output = cv2.resize(face, (output_size, output_size))
            image_output_rgb = cv2.cvtColor(image_output, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(output_folder, file), image_output_rgb)


        


if __name__ == '__main__':
    mtcnn_detect(here("data"), output_size=160)
