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
    # check dataset, directly delete unqualified images in dataset
    image_cnt = 0
    del_image_cnt = 0

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
            num_faces = det_res.shape[0]
            if num_faces != 1:
                os.remove(img_path)
                del_image_cnt += 1
        # check if this person has less than 2 images
        image_list = os.listdir(pp)
        if len(image_list) < 2:
            for img_file in image_list:
                os.remove(os.path.join(pp, img_file))
                del_image_cnt += 1
            os.rmdir(pp)

    print(f'Process finished.\n'
          f'Totally {person_cnt} persons and {image_cnt} images, '
          f'removed {del_image_cnt} images due to no face or more than 1 faces.')


if __name__ == '__main__':
    main()
