import os
import pandas as pd
from ShadeRecommender import ShadeRecommender

class UserHandler():
    def __init__(self):
        self.virtualtryon_playground_path = os.path.join('static', 'playground', 'virtualtryon')
        self.lipshadefinder_playground_path = os.path.join('static', 'playground', 'lipshadefinder')
        self.user_generated_masks_path = os.path.join('static', 'user_generated_masks')
        self.loreals_df = pd.read_excel('./static/lipshades.xlsx')
        self.uploaded_lipshadefinder_filename = None
        self.uploaded_virtualtryon_filename = None
        self.shaderecommender = None
        print("========= Starting UserHandler =========")

    def get_uploaded_lipshadefinder_filename_path(self):
        return os.path.join(self.lipshadefinder_playground_path, self.uploaded_lipshadefinder_filename)
    
    def set_uploaded_lipshadefinder_filename(self, filename):
        self.uploaded_lipshadefinder_filename = filename
        self.shaderecommender = ShadeRecommender(self.get_uploaded_lipshadefinder_filename_path())
        print(f"Initialised ShadeRecommender on {filename}")
        pass
        
    