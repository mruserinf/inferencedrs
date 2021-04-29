import os
from score import Score

if __name__ == "__main__":
    score = Score()
    score.load()
    prediction_input_list = []
    with open(os.path.join(os.environ['DATA_PATH'], 'input.csv')) as input_fd:
        prediction_input_list = input_fd.readlines()

    with open(os.path.join(os.environ['OUTPUT_PATH'], 'output.csv'), 'w') as output_fd:
        for prediction_input in prediction_input_list:
            prediction_obj = [{'body': prediction_input.encode('utf-8')}]
            prediction = score.predict(prediction_obj)
            output_fd.write(str(prediction))
            output_fd.write('\n')
