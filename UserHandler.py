import os
import pandas as pd
from ShadeRecommender import ShadeRecommender
from LipColorizer import LipColorizer
from FaceCounter import FaceCounter

class UserHandler():
    def __init__(self):
        self.virtualtryon_playground_path = os.path.join('static', 'playground', 'virtualtryon')
        self.lipshadefinder_playground_path = os.path.join('static', 'playground', 'lipshadefinder')
        self.user_generated_masks_path = os.path.join('static', 'user_generated_masks')
        self.loreals_df = pd.read_excel('./static/lipshades.xlsx')
        self.uploaded_lipshadefinder_filename = None
        self.uploaded_virtualtryon_filename = None
        self.shaderecommender = None
        self.lipcolorizer = None
        self.facecounter = None
        self.loreal_df = pd.read_excel('./static/lipshades.xlsx')
        print("========= Starting UserHandler =========")

    def get_uploaded_lipshadefinder_filename_path(self, filename):
        self.uploaded_lipshadefinder_filename = filename
        return os.path.join(self.lipshadefinder_playground_path, self.uploaded_lipshadefinder_filename)
    
    def set_uploaded_lipshadefinder_filename(self):
        self.shaderecommender = ShadeRecommender(self.get_uploaded_lipshadefinder_filename_path(self.uploaded_lipshadefinder_filename))
        print(f"Initialised ShadeRecommender on {self.uploaded_lipshadefinder_filename}")
        pass
    
    #=========== Virtual Try On ===========
    def get_uploaded_virtualtryon_filename_path(self, filename):
        self.uploaded_virtualtryon_filename = filename
        return os.path.join(self.virtualtryon_playground_path, self.uploaded_virtualtryon_filename)
    
    def set_uploaded_virtualtryon_filename(self):
        self.lipcolorizer = LipColorizer(self.get_uploaded_virtualtryon_filename_path(self.uploaded_virtualtryon_filename))
        print(f"Initialised LipColorizer on {self.uploaded_virtualtryon_filename}")
        pass
    
    def return_facecounter_filename(self):
        return self.facecounter.filename
    
    def check_for_one_face(self, file_path):
        self.facecounter = FaceCounter(file_path)
        self.facecounter.save_labelled_faces_img()
        if self.facecounter.num_faces_detected == 1:
            return True
        else:
            return False
        
    
        