import cv2
import os
from retinaface import RetinaFace

class FaceCounter():
    def __init__(self, img_path):
        self.img_path  = img_path
        self.resp = self.detect_num_faces()
        self.num_faces_detected = len(self.detect_num_faces()) #frontend to call this attribute to check how many faces detected in the image

    def detect_num_faces(self):
        '''
        function returns dic output of the number of faces detected in the image
        '''
        resp = RetinaFace.detect_faces(self.img_path)
        return resp
    
    def save_labelled_faces_img(self):
        '''
        saved labelled image into directory
        '''
        # Loop through the detected faces and draw bounding boxes
        image = cv2.imread(self.img_path)
        for face_name, face_data in self.resp.items():
            x1, y1, x2, y2 = face_data['facial_area']

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Green color

            # Add a label with the face name and score
            label = f"{face_name}"
            cv2.putText(image, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Save the image with bounding boxes and labels
        filename = os.path.basename(self.img_path)
        
        directory_path = "./static/face_counter_labelled_images"

        # Create the new filepath
        labelled_img_path = os.path.join(directory_path, filename)

        cv2.imwrite(labelled_img_path, image)        

# test=FaceCounter("hehe.jpeg")
# print(test.num_faces_detected)
# test.save_labelled_faces_img()