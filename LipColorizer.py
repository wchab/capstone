import cv2
import numpy as np
import dlib
import os
from PIL import ImageColor

class LipColorizer:
    def __init__(self, shape_predictor_path, image_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        self.image = cv2.imread(image_path)
        self.image = cv2.resize(self.image, (0, 0), None, 1, 1)
        self.imgOriginal = self.image.copy()

    def empty(self, a):
        pass

    def create_box(self, img, points, scale=5, masked=False, cropped=True):
        if masked:
            mask = np.zeros_like(img)
            mask = cv2.fillPoly(mask, [points], (255, 255, 255))
            img = cv2.bitwise_and(img, mask)
        if cropped:
            bbox = cv2.boundingRect(points)
            x, y, w, h = bbox
            img_crop = img[y:y+h, x:x+w]
            cv2.resize(img_crop, (0, 0), None, scale, scale)
            return img_crop
        else:
            return mask

    def colorize_lips(self, hexcode):
        rgb = ImageColor.getcolor(hexcode, "RGB")
        r, g, b= rgb[0], rgb[1], rgb[2]
        imgGray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(imgGray)

        for face in faces:
            landmarks = self.predictor(imgGray, face)
            myPoints = []
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                myPoints.append([x, y])

            myPoints = np.array(myPoints)
            imgLips = self.create_box(self.image, myPoints[48:61], 3, masked=True, cropped=False)

            imgColorLips = np.zeros_like(imgLips)
            imgColorLips[:] = b, g, r # 153, 0, 157
            imgColorLips = cv2.bitwise_and(imgLips, imgColorLips)
            imgColorLips = cv2.GaussianBlur(imgColorLips, (7, 7), 10)
            imgColorLips = cv2.addWeighted(self.imgOriginal, 1, imgColorLips, 0.4, 0)

            return imgColorLips
        
    def saveImage(self, colored_image, image_path):
        image_name, image_ext = os.path.splitext(image_path)
        new_name = image_name + image_ext
        cv2.imwrite(new_name, colored_image)

if __name__ == "__main__":
    shape_predictor_path = "./static/shape_predictor_68_face_landmarks.dat"
    image_path = "./static/playground/jennie.jpg"

    lip_colorizer = LipColorizer(shape_predictor_path, image_path)
    colored_image = lip_colorizer.colorize_lips('#781C44')

    cv2.imshow("Colored", colored_image)
    lip_colorizer.saveImage(colored_image, image_path)
    cv2.waitKey(0)
