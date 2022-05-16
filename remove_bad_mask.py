"""
(For masked dataset) Remove masked images which has 0 or more than 1 faces detected.
"""

import os
import yaml
import cv2
from tqdm import tqdm

from face_sdk.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from face_sdk.core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler

ds_dir = '../../cv_dataset/msra_tiny'


def remove_bad_mask():
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
        pp_unmasked = os.path.join(pp, 'unmasked')
        image_list = os.listdir(pp_unmasked)
        image_cnt += len(image_list)
        for img_file in image_list:
            dot = img_file.rfind('.')
            masked_img_file = img_file[:dot] + '-masked' + img_file[dot:]
            img_path = os.path.join(pp_unmasked, img_file)
            masked_img_path = os.path.join(pp_masked, masked_img_file)
            image = cv2.imread(masked_img_path)
            det_res = face_det_model_handler.inference_on_image(image)
            num_faces = det_res.shape[0]
            if num_faces == 0:
                missing_face_cnt += 1
            if num_faces > 1:
                redundant_face_cnt += 1
            if num_faces != 1:
                print(f'Removing {img_path} and its masked copy: {num_faces} faces.')
                os.remove(img_path)
                os.remove(masked_img_path)

    print(f'Process finished.\n'
          f'Totally {image_cnt} images, '
          f'where {missing_face_cnt} missing faces and {redundant_face_cnt} faces were removed '
          f'(totally {(missing_face_cnt + redundant_face_cnt) * 2} images).')


if __name__ == '__main__':
    remove_bad_mask()
