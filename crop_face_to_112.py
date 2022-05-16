"""
Crop face images in masked train set, save 224x224 results.
"""

import os
import yaml
import numpy as np
import cv2
from skimage import io as skio
from tqdm import tqdm

from face_sdk.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from face_sdk.core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from face_sdk.core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from face_sdk.core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from face_sdk.core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper

ds_dir = 'd:/my_document/cv_dataset/msra_tiny'
# ds_dir = 'test_dataset'


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

    print('Loading face alignment model ...')
    model_path = 'models'
    scene = 'non-mask'
    model_category = 'face_alignment'
    model_name = model_cfg[scene][model_category]
    face_align_model_loader = FaceAlignModelLoader(model_path, model_category, model_name)
    model, cfg = face_align_model_loader.load_model()
    face_align_model_handler = FaceAlignModelHandler(model, 'cuda:0', cfg)

    face_cropper = FaceRecImageCropper()

    # crop face images
    print('Cropping face images ...')
    image_cnt = 0
    no_face_cnt = 0

    person_list = os.listdir(ds_dir)
    person_cnt = len(person_list)
    for p in tqdm(person_list):
        pp = os.path.join(ds_dir, p)
        pp_masked = os.path.join(pp, 'masked')
        pp_unmasked = os.path.join(pp, 'unmasked')

        for img_dir in [pp_unmasked, pp_masked]:
            image_list = os.listdir(img_dir)
            image_cnt += len(image_list)
            for img_file in image_list:
                img_path = os.path.join(img_dir, img_file)
                image = skio.imread(img_path)
                if image.shape[0] == 112 and image.shape[1] == 112:  # already cropped
                    continue

                det_res = face_det_model_handler.inference_on_image(image)
                det_res = det_res.tolist()
                if len(det_res) == 0:
                    no_face_cnt += 1
                    continue

                # crop the most probable face
                det_res.sort(key=lambda x: x[-1], reverse=True)
                first_det_res = np.asarray(det_res[0][0:4], dtype=np.int32)
                align_res = face_align_model_handler.inference_on_image(image, first_det_res)
                align_res_list = align_res.flatten().tolist()

                cv2_image = image[..., ::-1].copy()
                cropped_image = face_cropper.crop_image_by_mat(cv2_image, align_res_list)
                cv2.imwrite(img_path, cropped_image)

    print(f'Process finished.\n'
          f'Totally {person_cnt} persons and {image_cnt} images, '
          f'{no_face_cnt} images not cropped because no face detected.\n')


if __name__ == '__main__':
    main()
