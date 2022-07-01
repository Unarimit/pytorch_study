from torchvision import transforms
from PIL import Image
import cv2
import os

if __name__ == '__main__':
    img_paths = []

    # add img paths
    PATH = '../division/data0'
    for _class in os.listdir(PATH):
        path2 = PATH + '/' + _class
        y_pred = None
        count = 1
        for imgf in os.listdir(path2):
            img_paths.append(path2 + '/' + imgf)

    transform = transforms.Compose([
                transforms.Resize((40, 40)),
                transforms.RandomChoice([
                    transforms.RandomRotation(degrees=(87.5, 92.5)),
                    transforms.RandomRotation(degrees=(-2.5, 2.5)),
                    transforms.RandomRotation(degrees=(-92.5, -87.5)),
                    transforms.RandomRotation(degrees=(-182.5, -177.5)),
                ]),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    for img_path in img_paths:
        img0 = Image.open(img_path)
        img = transform(img0)
        img.show()
        input()
