"""
Check whether masked and unmasked images align
"""

import os
import yaml
import cv2
from tqdm import tqdm

from face_sdk.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from face_sdk.core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler

ds_dir = '../../cv_dataset/msra_tiny'


def check_missing(delete=False):
    person_list = os.listdir(ds_dir)
    missing_img = 0
    missing_dir = 0
    for p in tqdm(person_list):
        pp = os.path.join(ds_dir, p)
        pp_masked = os.path.join(pp, 'masked')
        pp_unmasked = os.path.join(pp, 'unmasked')
        if not os.path.exists(pp_masked) or not os.path.exists(pp_unmasked):
            print(f'ERROR: masked or unmasked dir not exist under {p}.')
            missing_dir += 1
            continue

        image_list = os.listdir(pp_unmasked)
        for img_file in image_list:
            dot = img_file.rfind('.')
            masked_file = img_file[:dot] + '-masked' + img_file[dot:]
            masked_path = os.path.join(pp_masked, masked_file)
            if not os.path.exists(masked_path):
                print(f'ERROR: masked image {masked_path} not exist.')
                missing_img += 1
                if delete:
                    unmasked_path = os.path.join(pp_unmasked, img_file)
                    os.remove(unmasked_path)

    print(f'Missing dir: {missing_dir}, missing image: {missing_img}')
    if delete and missing_img > 0:
        print('Removed images whose masked image is missing.')


def check_face_detection():
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
    # check if there are masked faces which cannot be detected
    image_cnt = 0
    missing_face_cnt = 0
    redundant_face_cnt = 0

    person_list = os.listdir(ds_dir)
    for p in tqdm(person_list):
        pp = os.path.join(ds_dir, p)
        pp_masked = os.path.join(pp, 'masked')
        image_list = os.listdir(pp_masked)
        image_cnt += len(image_list)
        for img_file in image_list:
            img_path = os.path.join(pp_masked, img_file)
            image = cv2.imread(img_path)
            det_res = face_det_model_handler.inference_on_image(image)
            num_faces = det_res.shape[0]
            if num_faces == 0:
                missing_face_cnt += 1
            if num_faces > 1:
                redundant_face_cnt += 1
                print(f'ERROR: ??? ({num_faces} faces in {img_path})')

    print(f'Process finished.\n'
          f'Totally {image_cnt} images, '
          f'{missing_face_cnt} missing faces due to masking.')
    if redundant_face_cnt > 0:
        print(f'However, there are still {redundant_face_cnt} images which have more than 1 faces.')


if __name__ == '__main__':
    check_missing()
