import os

for path, subdirs, files in os.walk('/opt/ml/processing'):
    for name in files:
        print(os.path.join(path, name))
        print('\n')
