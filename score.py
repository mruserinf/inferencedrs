import json
import pandas as pd
import pickle
import os
import glob
from io import StringIO
import time


class Score:
    """
    A sample DRS Model handler implementation.
    """
    # Function to get prediction

    def predict(self, input):

        print(input)
        scaler = self.scaler

        # Generate Input
        processed_input = self.generate_input(input, scaler)
        
        # Predict & Return
        return [self.drs_model.predict_proba(processed_input)[0, 1]]

    # Function to get the input

    def generate_input(self, input, scaler):
        # Read Input
        # Input = pd.read_csv(Input_file)
        input_data = input[0]

        print(input_data)

        print(input_data['body'])

        body = input_data['body']
        body_string = 'sao2_cov,sao2_avg,apachescore,age,unitdischargeoffset,Hospital_LOS,heartrate_avg,respiration_avg,respiration_cov,BMI,heartrate_cov,WBC x 1000_max,bedside glucose_min,creatinine_max,sodium_max,heartrate_max,FiO2_count,heartrate_min,bedside glucose_max,WBC x 1000_min,potassium_max,pH_count,sodium_avg,creatinine_min,potassium_min\n' + \
            body.decode('utf-8')

        input = pd.read_csv(StringIO(body_string), delimiter=',')

        print(input)

        order = ['sao2_cov', 'sao2_avg', 'apachescore', 'age', 'unitdischargeoffset', 'Hospital_LOS', 'heartrate_avg',
                 'respiration_avg', 'respiration_cov', 'BMI', 'heartrate_cov', 'WBC x 1000_max', 'bedside glucose_min',
                 'creatinine_max', 'sodium_max', 'heartrate_max', 'FiO2_count', 'heartrate_min', 'bedside glucose_max',
                 'WBC x 1000_min', 'potassium_max', 'pH_count', 'sodium_avg', 'creatinine_min', 'potassium_min']
        input = input[order]
        # Get in right order
        input = scaler.transform(input)
        # print(Input)

        # Return
        return input

    def load(self):
        print('load_model:*************')
        # Read Model & scaler
        files = pickle.load(open('DRS_Model.pl', 'rb'), encoding="bytes")
        print(input)
        self.drs_model = files['model']
        self.scaler = files['scaler']
        return True