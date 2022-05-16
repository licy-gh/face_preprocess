"""
MS-Celeb dataset: 86,876 ids, 3,923,399 pics, 125.30 GiB.
"""

import os
import sys
from tqdm import tqdm
import math
import random

data_dir = 'd:/my_document/cv_dataset/msra'
new_dir = 'd:/my_document/cv_dataset/msra_tiny'


def data_statistics(new=False):
    target_dir = new_dir if new else data_dir
    dirs = os.listdir(target_dir)
    print(f'Totally {len(dirs)} ids.')

    pic_num = 0
    total_size = 0
    for d in tqdm(dirs):
        id_dir = os.path.join(target_dir, d)
        pics = os.listdir(id_dir)
        pic_num += len(pics)
        for pic in pics:
            total_size += os.path.getsize(os.path.join(id_dir, pic))
    total_size /= math.pow(2, 30)
    print(f'Totally {pic_num} pics.')
    print(f'Total size: {"%.3f" % total_size} GiB.')


def data_statistics_masked():
    target_dir = new_dir
    dirs = os.listdir(target_dir)
    print(f'Totally {len(dirs)} ids.')

    pic_num = 0
    total_size = 0
    for d in tqdm(dirs):
        id_dir = os.path.join(target_dir, d)
        id_dir_masked = os.path.join(id_dir, 'masked')
        id_dir_unmasked = os.path.join(id_dir, 'unmasked')
        pics_masked = os.listdir(id_dir_masked)
        pics_unmasked = os.listdir(id_dir_unmasked)
        pic_num += (len(pics_masked) + len(pics_unmasked))
        for pic in pics_masked:
            total_size += os.path.getsize(os.path.join(id_dir_masked, pic))
        for pic in pics_unmasked:
            total_size += os.path.getsize(os.path.join(id_dir_unmasked, pic))
    total_size /= math.pow(2, 20)
    print(f'Totally {pic_num} pics.')
    print(f'Total size: {"%.3f" % total_size} MiB.')


def select(p: float = 0.01):
    p = max(min(p, 1), 0)

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    dirs = os.listdir(data_dir)
    num_ids = len(dirs)
    new_num_ids = int(num_ids * p)
    new_idx = random.sample(list(range(num_ids)), new_num_ids)

    print(f'Creating a subset with {new_num_ids} ids ...')
    for i, d in tqdm(enumerate(dirs)):
        if i in new_idx:
            id_dir = os.path.join(data_dir, d)
            new_id_dir = os.path.join(new_dir, d)
            pics = os.listdir(id_dir)
            if not os.path.exists(new_id_dir):
                os.makedirs(new_id_dir)

            for pic in pics:
                f = open(os.path.join(id_dir, pic), 'rb')
                nf = open(os.path.join(new_id_dir, pic), 'wb')
                data = f.read()
                nf.write(data)
                f.close()
                nf.close()

    print('Finished creating subset.')


def select_alpha(num_pic: int, alpha: float = 0.5):
    alpha = max(min(alpha, 1), 0)

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    dirs = os.listdir(data_dir)
    num_ids = len(dirs)
    id_list = list(range(num_ids))
    random.shuffle(id_list)

    num_selected_pic = 0
    new_num_ids = 0

    print(f'Selecting {num_pic} pictures from the dataset ...')
    progress_bar(num_selected_pic, num_pic)
    for i in id_list:
        d = dirs[i]
        id_dir = os.path.join(data_dir, d)
        new_id_dir = os.path.join(new_dir, d)
        pics = os.listdir(id_dir)

        id_num_pic = int(alpha * len(pics))
        if id_num_pic < 2:
            continue

        if not os.path.exists(new_id_dir):
            os.makedirs(new_id_dir)

        for pi in range(id_num_pic):
            pic = pics[pi]
            f = open(os.path.join(id_dir, pic), 'rb')
            nf = open(os.path.join(new_id_dir, pic), 'wb')
            data = f.read()
            nf.write(data)
            f.close()
            nf.close()

        num_selected_pic += id_num_pic
        new_num_ids += 1
        progress_bar(num_selected_pic, num_pic)
        if num_selected_pic >= num_pic:
            break

    print('\n')
    if num_selected_pic < num_pic:
        print(f'Warning: Dataset exhausted but failed to '
              f'collect enough pictures ({num_selected_pic}/{num_pic}).')
    print(f'Finished creating a subset with {new_num_ids} ids, {num_selected_pic} pictures.')


def progress_bar(cur: int, total: int):
    percentage = int(cur * 100 / total)
    bn = percentage // 2
    wn = 50 - bn
    print(f'\r{"%2d" % percentage}%|' + '▉' * bn + ' ' * wn + '|' + f'{cur}/{total}', end='')
    sys.stdout.flush()


if __name__ == '__main__':
    data_statistics_masked()
