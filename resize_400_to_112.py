"""
Resize 400x400 images to 112x112 images to unify the train set.
"""

import os
import cv2
from tqdm import tqdm

ds_dir = 'd:/my_document/cv_dataset/msra_tiny'


def main():
    print('Resizing 400x400 face images to 112x112 ...')
    image_cnt = 0
    resize_cnt = 0

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
                image = cv2.imread(img_path)
                if image.shape[0] == 400 and image.shape[1] == 400:
                    image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(img_path, image)
                    resize_cnt += 1

    print(f'Process finished.\n'
          f'Totally {person_cnt} persons and {image_cnt} images, '
          f'{resize_cnt} images resized to 112x112.\n')


if __name__ == '__main__':
    main()
