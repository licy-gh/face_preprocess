import os
import pickle
import yaml
import imghdr
from skimage.io import imread, imsave
import numpy as np
import random
from tqdm import tqdm

from face_sdk.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from face_sdk.core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from face_sdk.core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from face_sdk.core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from add_mask.face_masker import FaceMasker

testset_dir = '../../cv_dataset/testsets'
mask_dir = 'add_mask/Data/mask-data'


def check_type(name: str):
    ds_name = name + '.bin'
    ds_path = os.path.join(testset_dir, ds_name)
    if not os.path.exists(ds_path):
        print(f'ERROR: dataset {name} not exist. Abort.')
        exit(0)

    f = open(ds_path, 'rb')
    d0, d1 = pickle.load(f, encoding='bytes')
    f.close()

    # check image type
    types = {}
    for i in range(len(d0)):
        ret = imghdr.what(None, bytes(d0[i]))
        if types.get(ret) is None:
            types[ret] = 0
        types[ret] += 1
    print('types:', types)


def make_testset(name: str):
    ds_name = name + '.bin'
    ds_path = os.path.join(testset_dir, ds_name)
    if not os.path.exists(ds_path):
        print(f'ERROR: dataset {name} not exist. Abort.')
        exit(0)

    ds_dir = os.path.join(testset_dir, name)
    if not os.path.exists(ds_dir):
        os.mkdir(ds_dir)
    true_dir = os.path.join(ds_dir, 'true')
    false_dir = os.path.join(ds_dir, 'false')
    if not os.path.exists(true_dir):
        os.mkdir(true_dir)
    if not os.path.exists(false_dir):
        os.mkdir(false_dir)

    f = open(ds_path, 'rb')
    d0, d1 = pickle.load(f, encoding='bytes')
    f.close()

    # extract images
    ti = 0
    fi = 0
    print('Extracting images ...')
    for i in tqdm(range(len(d1))):
        subdir = 'true' if d1[i] else 'false'
        ii = ti if d1[i] else fi
        f = open(os.path.join(ds_dir, subdir, f'{ii}au.jpeg'), 'wb')
        f.write(bytes(d0[2 * i]))
        f.close()
        f = open(os.path.join(ds_dir, subdir, f'{ii}bu.jpeg'), 'wb')
        f.write(bytes(d0[2 * i + 1]))
        f.close()
        if d1[i]:
            ti += 1
        else:
            fi += 1

    # load model for adding masks
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

    mask_list = os.listdir(mask_dir)
    add_mask_cnt = [0] * len(mask_list)

    print('Loading face masker ...')
    face_masker = FaceMasker(is_aug=False)

    print('Adding masks for dataset ...')
    for subdir in ['true', 'false']:
        image_dir = os.path.join(ds_dir, subdir)
        image_list = os.listdir(image_dir)
        for img_file in tqdm(image_list):
            img_path = os.path.join(image_dir, img_file)
            image = imread(img_path)
            det_res = face_det_model_handler.inference_on_image(image)
            det_res = det_res.tolist()
            if len(det_res) == 0:
                continue

            # add mask for the most probable face
            det_res.sort(key=lambda x: x[-1], reverse=True)
            first_det_res = np.asarray(det_res[0][0:4], dtype=np.int32)
            align_res = face_align_model_handler.inference_on_image(image, first_det_res)
            align_res_list = align_res.flatten().tolist()

            # randomly choose a mask
            mask_idx = random.randint(0, len(mask_list) - 1)
            mask_name = mask_list[mask_idx]
            masked_image = face_masker.add_mask_one(image, align_res_list, mask_name)
            masked_image = masked_image * 255
            masked_image = masked_image.astype(np.uint8)

            masked_file = img_file.replace('u.', 'm.')
            save_path = os.path.join(image_dir, masked_file)
            imsave(save_path, masked_image)
            add_mask_cnt[mask_idx] += 1

    print(f'Added masks for {sum(add_mask_cnt)} images.\n'
          f'Mask type distribution: {add_mask_cnt}.')

    print(f'Making database finished.')


if __name__ == '__main__':
    check_type('calfw')
    make_testset('calfw')
