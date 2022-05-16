import os
import yaml
import cv2
from tqdm import tqdm

from face_sdk.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from face_sdk.core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler

ds_dir = 'd:/my_document/cv_dataset/msra_tiny'


def main():
    # load model
    f = open('face_sdk/config/model_conf.yaml', 'r')
    model_cfg = yaml.load(f, yaml.SafeLoader)

    print('Loading face detection model ...')
    model_path = 'models'
    scene = 'non-mask'
    model_category = 'face_detection'
    model_name = model_cfg[scene][model_category]
    face_det_model_loader = FaceDetModelLoader(model_path, model_category, model_name)
    model, cfg = face_det_model_loader.load_model()
    face_det_model_handler = FaceDetModelHandler(model, 'cuda:0', cfg)

    print('Checking dataset ...')
    # check face numbers of images dataset
    image_cnt = 0
    face_num = [0] * 5

    person_list = os.listdir(ds_dir)
    person_cnt = len(person_list)
    for p in tqdm(person_list):
        pp = os.path.join(ds_dir, p)
        image_list = os.listdir(pp)
        image_cnt += len(image_list)
        for img_file in image_list:
            img_path = os.path.join(pp, img_file)
            image = cv2.imread(img_path)
            det_res = face_det_model_handler.inference_on_image(image)
            fn = det_res.shape[0]
            if fn <= 3:
                face_num[fn] += 1
            else:
                face_num[-1] += 1

    print(f'Process finished.\n'
          f'Totally {person_cnt} persons and {image_cnt} images in dataset.\n'
          f'Face number distribution:\n([0, 1, 2, 3, >3])\n{face_num}.')


if __name__ == '__main__':
    main()
