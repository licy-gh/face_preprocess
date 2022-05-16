import torch
import pickle

model_path = 'models/face_detection/face_detection_1.0/face_detection_retina.pkl'


if __name__ == '__main__':
    data = torch.load(model_path)

    print(data)
    print(type(data))
