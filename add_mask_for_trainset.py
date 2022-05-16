import os
import yaml
import numpy as np
from skimage.io import imread, imsave
import random
from tqdm import tqdm

from face_sdk.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from face_sdk.core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from face_sdk.core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from face_sdk.core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from add_mask.face_masker import FaceMasker

ds_dir = 'test_dataset'
mask_dir = 'add_mask/Data/mask-data'


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

    print('Loading face masker ...')
    face_masker = FaceMasker(is_aug=False)

    print('Adding masks for dataset ...')
    # add masks for dataset, put masked and unmasked faces into two directories
    image_cnt = 0
    mask_list = os.listdir(mask_dir)
    add_mask_cnt = [0] * len(mask_list)

    person_list = os.listdir(ds_dir)
    person_cnt = len(person_list)
    for p in tqdm(person_list):
        pp = os.path.join(ds_dir, p)
        image_list = os.listdir(pp)
        if 'masked' in image_list:
            image_list.remove('masked')
        if 'unmasked' in image_list:
            image_list.remove('unmasked')

        # arrange unmasked images
        pp_masked = os.path.join(pp, 'masked')
        pp_unmasked = os.path.join(pp, 'unmasked')
        if os.path.exists(pp_masked) and os.path.exists(pp_unmasked) \
                and len(os.listdir(pp_masked)) == len(os.listdir(pp_unmasked)):
            image_cnt += len(os.listdir(pp_unmasked))
            continue
        if not os.path.exists(pp_masked):
            os.makedirs(pp_masked)
        if not os.path.exists(pp_unmasked):
            os.makedirs(pp_unmasked)
        for img_file in image_list:
            img_path = os.path.join(pp, img_file)
            f = open(img_path, 'rb')
            data = f.read()
            f.close()
            f = open(os.path.join(pp_unmasked, img_file), 'wb')
            f.write(data)
            f.close()
            os.remove(img_path)

        # generate masked images
        image_list = os.listdir(pp_unmasked)
        image_cnt += len(image_list)
        for img_file in image_list:
            img_path = os.path.join(pp_unmasked, img_file)
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

            dot = img_file.rfind('.')
            masked_file = img_file[:dot] + '-masked' + img_file[dot:]
            save_path = os.path.join(pp_masked, masked_file)
            imsave(save_path, masked_image)
            add_mask_cnt[mask_idx] += 1

    print(f'Process finished.\n'
          f'Totally {person_cnt} persons and {image_cnt} images, added masks for {sum(add_mask_cnt)} images.\n'
          f'Mask type distribution: {add_mask_cnt}.')


if __name__ == '__main__':
    main()
