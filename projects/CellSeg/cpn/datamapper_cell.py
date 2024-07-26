import this
import scipy.io
import scipy.ndimage
import os
import os.path as osp
import json
import math
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import SizeMismatchError
from detectron2.structures import BoxMode
from PIL import Image
import cv2
import numpy as np
import random
import pickle
import torch
import matplotlib.pyplot as plt
from os.path import splitext


from .datasets.cpn import CPNTargetGenerator
from .datasets import cpn


class CellDatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        print("for cell")
        print(cfg.all_iter, cfg.e_sche, cfg.samples, cfg.img_file, cfg.data_file, cfg.transforms, cfg.item)
        # Rebuild augmentations
        # print(cfg)
        # print(cfg.items)
        self.items = None if cfg.item <= 0 else cfg.item
        # print(self.items)
        # assert 1 == 0
        self.split = 'train' if is_train else 'test'
        # self.split = 'train'
        # self.split = 'val'
        # self.split = 'train'
        self.training = is_train

        self.is_cos = cfg.is_cos
        self.use_freq_guild = cfg.use_freq_guild

        print("训练模式") if is_train else print("测试模式")

        if not self.is_cos:
            legal_img_list = pickle.load(
                open(
                    "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/合并标注数据/data_512_new/legal_" + self.split + "_files.pkl",
                    'rb')
            )
            clegal_img_list = []
            for e in legal_img_list:
                if e == 'dfa69dce-d7e7-11ec-a8b3-d562838f25df_2.bmp':
                    # print("dddd")
                    # print(e)
                    # assert 1 == 0
                    e = splitext(e)[0] + '.png'
                # else:
                clegal_img_list.append(e)

            self.legal_img_list = legal_img_list

        else:
            img_files = "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/new_seg_data/jzx/"
            # img_files = "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/single_test"
            legal_img_list = os.listdir(img_files)  # ori_imgs
            cimg_names = []
            for e in legal_img_list:
                if e.split(".")[-1] not in ['png', 'bmp', 'jpg']:
                    # print(e)
                    continue
                cimg_names.append(e)

            img_files = "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/new_seg_data/tct/"
            legal_img_list = os.listdir(img_files)  # ori_imgs
            for e in legal_img_list:
                if e.split(".")[-1] not in ['png', 'bmp', 'jpg']:
                    # print(e)
                    continue
                cimg_names.append(e)

            self.legal_img_list = cimg_names
            # print(self.legal_img_list)
            # assert 1 == 0

        self.items = len(self.legal_img_list)

        self.gen = CPNTargetGenerator(
            samples=cfg.samples,
            order=cfg.order,
            min_fg_dist=cfg.min_fg_dist,
            max_bg_dist=cfg.max_bg_dist
        )
        self.transforms = None if cfg.transforms == 0 else cfg.transforms
        self.data_file = cfg.data_file
        self.img_root = cfg.img_file
        self.mask_root = cfg.mask_root
        self.height = 512
        self.width = 512

        self.data_name = cfg.data_name
        # if not self.is_cos and not (self.data_name in ['dsb'] and self.split=='test'):
        if not self.is_cos:
            self.load_data()

    def load_data_for_dsb(self, file_root, img_name):
        print("#################")
        print(img_name)
        ori_img = cv2.imread(os.path.join(file_root, splitext(img_name)[0], 'images', img_name), 3)  # BGR
        height, width = ori_img.shape[:2]
        if height != self.height or width != self.width:

            ramdon = False
            if ramdon:
                print("random clip (" + str(height) + ", " + str(width) + ") to (512, 512)")
                start_x = random.randint(0, height - self.height)
                start_y = random.randint(0, width - self.width)
                ori_img = ori_img[start_x: start_x + self.height, start_y: start_y + self.width, :]
            else:  # resize
                print("resize (" + str(height) + ", " + str(width) + ") to (512, 512)")
                dy = float(self.height) / height
                dx = float(self.width) / width

                ori_img = cv2.resize(ori_img, dsize=None, fx=dx, fy=dy, interpolation=cv2.INTER_LINEAR)

        height, width = ori_img.shape[:2]
        assert height == self.height and width == self.width, ori_img.shape
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)  # convert to RGB

        return {
            'image_name': img_name,
            'image': ori_img,
            'height': self.height,
            'width': self.width,

        }


    def load_data_for_cos(self, file_root, img_name):
        print("#################")
        print(img_name)
        ori_img = cv2.imread(os.path.join(file_root, img_name), 3)  # BGR

        height, width = ori_img.shape[:2]
        if height != 512 or width != 512:

            ramdon = False
            if ramdon:
                print("random clip (" + str(height) + ", " + str(width) + ") to (512, 512)")
                start_x = random.randint(0, height - 512)
                start_y = random.randint(0, width - 512)
                ori_img = ori_img[start_x: start_x + 512, start_y: start_y + 512, :]
            else: # resize
                print("resize (" + str(height) + ", " + str(width) + ") to (512, 512)")
                dy = float(self.height)/height
                dx = float(self.width)/width

                ori_img = cv2.resize(ori_img, dsize=None, fx=dx, fy=dy, interpolation=cv2.INTER_LINEAR)


            # cv2.imshow('ori img', ori_img)
            # cv2.waitKey(0)

        # continue

        height, width = ori_img.shape[:2]
        assert height == self.height and width == self.width, ori_img.shape
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)  # convert to RGB

        return {
            'image_name': img_name,
            'image': ori_img,
            'height': self.height,
            'width': self.width,

        }

    def ranodm_clip(self, img, mask, scale=0.5):
        height, width = int(img.shape[0]*scale), int(img.shape[1]*scale)

        x = random.randint(0, img.shape[0]-height)
        y = random.randint(0, img.shape[1]-width)

        cropped = img[y: y+height, x: x+width]
        cropped_mask = mask[y: y+height, x: x+width]
        print(cropped_mask.shape, cropped.shape)
        resized = cv2.resize(cropped, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(cropped_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

        return resized, resized_mask

    def colorjitter(self, img, cj_type='b'):
        '''
        cj_type {'b': brightness, 's': saturation, 'c': constast}
        '''
        oimg = img.copy()
        if cj_type == 'b':
            value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            h, s, v = cv2.split(hsv)

            if value > 0:
                lim = 255 - value
                v[v > lim] = 255
                v[v <= lim] += value
            else:
                lim = np.absolute(value)
                v[v < lim] = 0
                v[v >= lim] -= np.absolute(value)

            final_hsv = cv2.merge((h, s, v))
            cimg = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            # return img

        elif cj_type == 's':
            value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            h, s, v = cv2.split(hsv)
            if value > 0:
                lim = 255 - value
                s[s > lim] = 255
                s[s <= lim] += value
            else:
                lim = np.absolute(value)
                s[s < lim] = 0
                s[s >= lim] -= np.absolute(value)

            final_hsv = cv2.merge((h, s, v))
            cimg = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            # return img

        elif cj_type == 'c':
            brightness = 10
            contrast = random.randint(40, 100)
            dummy = np.int16(img)
            dummy = dummy * (contrast/127+1) - contrast + brightness
            dummy = np.clip(dummy, 0, 255)
            cimg = np.uint8(dummy)

        else:
            raise

        return cimg

    def rotate(self, img, mask):
        height, width = img.shape[0], img.shape[1]

        centerx = random.randint(int(height//10), int(height//1.2))
        centery = random.randint(int(width//10), int(width//1.2))

        degree = random.random()*180

        M = cv2.getRotationMatrix2D((centery, centerx), degree, 1)
        dst = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        dst_mask = cv2.warpAffine(mask, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0))

        return dst, dst_mask


    def augment_hsv(self, img, hgain=0.015, sgian=0.7, vgain=0.4):
        r = np.random.uniform(-1, 1, 3) * [hgain, sgian, vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

        dtype = img.dtype

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 100).astype(dtype)

        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)

        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


    def load_data(self):
        assert self.data_file is not None
        datas = json.load(open(self.data_file, 'r'))

        # if not self.training:
        #     print("random shuffle to construct test datasets...")
        # random.shuffle(datas)

        print("number of dataset: ", len(datas))

        datasets = {}

        for i in range(len(datas)):
            if self.items is not None and len(datasets) >= self.items:
                break
            # assert 1 == 0
            e = datas[i]
            if e['img_name'] not in self.legal_img_list and self.data_name == 'our':
                continue
            if self.data_name=='our' and \
                e['img_name'] in ['TC19008021_13940_8923_1274_1079.png', 'TC19008200_34607_40749_1216_1036.png', 'TC19008200_46727_26557_1266_1225.png', 'TC19009815_10871_27086_1261_1186.png', 'TC19008175_29692_17883_1400_1119.png', 'TC19008234_42265_30573_1349_1180.png', 'TC19009820_34389_18968_1075_1044.png', 'TC19008021_27760_29315_1149_1190.png', 'TC19009815_22369_42838_1412_1093.png', 'TC19009820_16251_19508_1141_1087.png', 'TC19024734_39438_24273_1113_1054.png', 'TC19008141_57171_38701_1399_1143.png', 'TC19008141_17983_30341_1380_1249.png', 'TC19008175_28071_26910_1243_1127.png', 'TC19009820_16992_25644_1083_1057.png', 'TC19008200_19951_25530_1126_1154.png', 'TC19008021_20500_44351_1157_1057.png', 'TC19008021_10482_25873_1170_1069.png', 'TC19024734_8283_31922_1306_1177.png', 'TC19024734_36345_36259_1354_1118.png', 'TC19008021_36749_35668_1205_1030.png', 'TC19024725_5721_32929_1131_1196.png', 'TC19008141_19166_18763_1358_1124.png', 'TC19008200_47878_43024_1322_1256.png', 'TC19008234_46937_54600_1186_1057.png', 'TC19008175_6086_28861_1151_1090.png', 'TC19024725_21896_22785_1396_1059.png', 'TC19024731_51780_36988_1390_1239.png', 'TC19008234_53711_16581_1290_1162.png', 'TC19024734_30497_6433_1197_1055.png', 'TC19008234_24889_7867_1629_1404.png', 'TC19008141_26820_8158_1261_1173.png', 'TC19008021_20024_7067_1152_1152.png', 'TC18006277_20932_27711_1923_1334.png', 'TC19009815_30304_34350_1354_1166.png', 'TC19008141_38066_23649_1308_1169.png', 'TC19024731_25187_55064_1372_1152.png', 'TC19008200_16538_37357_1202_1151.png', 'TC19024725_40491_10961_1157_1229.png', 'TC19009815_28977_11203_1125_1092.png', 'TC19008200_16678_43315_1314_1137.png', 'TC19008141_45298_43435_1161_1070.png', 'TC19024734_10771_13844_1173_1088.png', 'TC19008175_24026_39538_1239_1067.png', 'TC19024734_9202_43289_1235_1122.png', 'TC19008175_16815_17238_1147_1119.png', 'TC19024731_40135_31235_1460_1250.png', 'TC19024725_36159_26968_1162_1228.png', 'TC19009815_39363_17452_1560_1258.png', 'TC19008175_50054_48782_1294_1174.png', 'TC19024731_37633_51682_1444_1434.png', 'TC19024731_28046_31793_1353_1238.png', 'TC19008175_11941_39796_1203_1083.png', 'TC19024734_11212_9806_1266_1078.png', 'TC19008021_17908_25809_1174_1028.png', 'TC19008141_37378_52308_1103_1050.png', 'TC19009820_28398_34115_1160_1042.png', 'TC19009815_51635_9005_1297_1125.png', 'TC19008200_31917_44872_1187_1041.png', 'TC19009820_30652_14279_1316_1172.png', 'TC19008234_16030_18864_1352_1263.png', 'TC19024725_41436_38136_1533_1199.png', 'TC19009820_43853_32345_1204_1135.png', 'TC19009820_36095_25223_1141_1125.png', 'TC19024725_10047_22197_1166_1069.png', 'TC19024731_63309_31387_1469_1315.png', 'TC19008200_48943_19828_1309_1053.png', 'TC19024734_34859_4520_1193_1097.png', 'TC19024725_21508_14475_1755_1173.png', 'TC19024734_47768_19753_1339_1148.png', 'TC19008175_34562_45789_1172_1049.png', 'TC19024725_22810_31843_1459_1221.png', 'TC19008234_25319_53082_1657_1325.png', 'TC19008021_11041_30352_1099_1195.png', 'TC19009815_47063_34647_1278_1103.png', 'TC19009815_14070_32383_1387_1225.png', 'TC19009815_37420_40492_1294_1198.png', 'TC19008200_53501_34385_1177_1180.png', 'TC19009820_30831_43563_1189_1026.png', 'TC19024725_39381_20163_1409_1166.png', 'TC19009820_17509_32322_1055_1036.png', 'TC19008021_20032_37784_1224_1047.png', 'TC19008021_26744_23401_1031_1054.png', 'TC19024731_31917_56081_1340_1250.png', 'TC19008141_27461_39241_1372_1230.png', 'TC19009820_27522_20659_1402_1203.png', 'TC19008234_27647_2871_1806_1583.png', 'TC19009815_18627_20217_1355_1363.png', 'TC19024731_14726_29897_1451_1462.png', 'TC19008175_47750_33437_1267_1103.png', 'TC19008200_26115_9846_1151_1031.png', 'TC19024731_54117_45548_1324_1321.png', 'TC19008234_11883_36123_1414_1267.png', 'TC19008141_25227_49724_1197_1097.png', 'TC19024734_20562_30199_1218_1120.png', 'TC19008175_46710_13160_1278_1086.png', 'TC19008141_46207_21427_1272_1139.png']:
                continue

            if e['img_name'] == 'dfa69dce-d7e7-11ec-a8b3-d562838f25df_2.bmp' and self.data_name == 'our':
                e['img_name'] = 'dfa69dce-d7e7-11ec-a8b3-d562838f25df_2.png'

            print(e.keys())  # dict_keys(['img_name', 'height', 'width', 'contours', 'category', 'includeIns'])
            img_name = e['img_name']
            # print(os.path.join(self.img_root, e['img_name']))
            img = cv2.imread(os.path.join(self.img_root, e['img_name']), 3)  # BGR
            if os.path.exists(os.path.join(self.mask_root, splitext(e['img_name'])[0]+'.png')):
                mask = cv2.imread(os.path.join(self.mask_root, splitext(e['img_name'])[0]+'.png'), 3)  # L
            elif os.path.exists(os.path.join(self.mask_root, splitext(e['img_name'])[0]+'.jpg')):
                mask = cv2.imread(os.path.join(self.mask_root, splitext(e['img_name'])[0]+'.jpg'), 3)  # L
            elif os.path.exists(os.path.join(self.mask_root, e['img_name'])):
                mask = cv2.imread(os.path.join(self.mask_root, e['img_name']), 3)  # L
            else:
                print(os.path.join(self.mask_root, e['img_name']))
                raise

            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # print(np.unique(mask))
            mask[mask<120] = 0# [0, 1]
            mask[mask!=0] = 1
            assert len(np.unique(mask).tolist()) <= 2, np.unique(mask)

            # guild_freq = self.generate_freq_guild(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

            # plt.subplot(141)
            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # plt.subplot(142)
            # plt.imshow(mask)
            # # plt.subplot(143)
            # # plt.imshow(guild_freq)
            # # plt.subplot(144)
            # # plt.imshow(np.clip(guild_freq+mask, a_max=1.0, a_min=0.0))
            # plt.show()
            # assert 1 == 0

            # print(os.path.join(self.img_root, e['img_name']))
            print(e['img_name'])
            print(img.shape)
            contours = e['contours']
            height = e['height']
            width = e['width']
            # print(height, width)
            assert img.shape[0] == self.height and img.shape[1] == self.width
            assert height == self.height and width == self.width

            # print(type(contours))
            contours_ = []
            print("number of contours: ", len(contours))
            print("include instance? ", e['includeIns'])
            print("category: ", e['category'])

            for el in contours:
                # print(type(el))
                # print(np.array(el).squeeze(1).shape)
                contours_.append(np.array(el).squeeze(1))  # [num_points, 1, 2] -> [num_points, 2]
                # assert 1 == 0
            # assert 1 == 0
            labels = cpn.contours2labels(contours_, (height, width))
            # print(type(labels))  # narray
            # print(labels.shape)  # [height, width, 3]
            # print(np.unique(labels))
            # cv2.imshow('ori img', labels)
            # cv2.waitKey(0)
            labels_ = cpn.resolve_label_channels(labels)
            #gen = self.gen
            #gen.feed(labels=labels_)
            #clabels_ = gen.reduced_labels
            #labels_ = clabels_
            #print(img.shape, mask.shape, labels_.shape)
            #plt.subplot(131)
            #plt.imshow(img)
            #plt.subplot(132)
            #plt.imshow(mask)
            #plt.subplot(133)
            #plt.imshow(labels_)
            #plt.show()
            #assert 1 == 0

            # print(labels_.dtype)


            # plt.subplot(131)
            # plt.imshow(img)
            #
            # plt.subplot(132)
            # plt.imshow(labels_)
            # plt.subplot(133)
            # plt.imshow(clabels_)
            #
            # plt.show()
            # plt.close()
            # assert 1 == 0
            # oimg = self.augment_hsv(img)
            # low_freq_guild1 = self.generate_freq_guild(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            # low_freq_guild2 = self.generate_freq_guild(cv2.cvtColor(oimg, cv2.COLOR_BGR2GRAY))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB

            # plt.subplot(1, 4, 1)
            # plt.imshow(img)
            # plt.subplot(1, 4, 2)
            # plt.imshow(low_freq_guild1)
            #
            # plt.subplot(1, 4, 3)
            # plt.imshow(oimg)
            # plt.subplot(1, 4, 4)
            # plt.imshow(low_freq_guild2)
            #
            # plt.show()
            # plt.close()
            # assert 1 == 0

            # plt.imshow(img)
            # plt.show()
            # plt.imshow(labels_)
            # plt.show()
            # plt.close()
            # assert 1 == 0

            datasets[e['img_name']] = {
                'image_name': e['img_name'],
                'image': img,
                'labels': labels_,
                'height': self.height,
                'width': self.width,
                'bmask': mask,

            }

        print("finish load dataset")
        # print(datasets.keys())
        # assert 1 == 0
        if self.items is None:
            self.items = len(datasets)

        self.datasets = datasets

    def __len__(self):
        return self.items

    @staticmethod
    def map(image):  # normalize pixel value
        image = image / 255
        return image

    @staticmethod
    def unmap(image):
        image = image * 255
        image = np.clip(image, 0, 255).astype('uint8')
        return image

    def datagenerate(self, index):
        # pass
        assert index in self.datasets.keys(), index + str(self.datasets.keys())

        return self.datasets[index]

    def generate_freq_guild(self, gray_img):

        dft = cv2.dft(np.float32(gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dftshift = np.fft.fftshift(dft)
        # remove low fre
        mask = np.ones(dftshift.shape, np.uint8)
        crow, ccol = int(dftshift.shape[0] / 2), int(dftshift.shape[1] / 2)
        # radius = int(0.0005*dftshift.shape[0]/2)
        radius = 1

        cv2.circle(mask, (crow, ccol), radius, 0, -1)
        low_shift = dftshift * mask

        low_ishift = np.fft.ifftshift(low_shift)
        lowiimg = cv2.idft(low_ishift)
        # res_low = 1000*np.log(cv2.magnitude(lowiimg[:, :, 0], lowiimg[:, :, 1]))
        res_low = 10000 * (cv2.magnitude(lowiimg[:, :, 0], lowiimg[:, :, 1]))
        low_freq_guild = res_low / float(np.max(res_low))  # [H, W]

        return low_freq_guild


    def __call__(self, dataset_dict):
        # print("call cell dataset mapper...")
        dataset_dicts = {}

        data_index = dataset_dict['data_index']
        data_key = dataset_dict['key']

        # if self.data_name == 'monuseg':
        #     return self.load_data_monuseg(data_key)

        if self.is_cos:
            file_root = dataset_dict['file_root']
            dataset = self.load_data_for_cos(file_root, data_key)
            img = dataset['image']

            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            if self.use_freq_guild:
                low_freq_guild = self.generate_freq_guild(gray_img)

            fgimg_mask = gray_img < 200  # black: True, white: False

            # normalize image
            img = self.map(img)

            dataset_dicts["width"] = self.width
            dataset_dicts["height"] = self.height

            dataset_dicts['image'] = torch.as_tensor(
                np.ascontiguousarray(img.transpose(2, 0, 1))
            )  # [3, H, W]

            dataset_dicts['file_name'] = dataset['image_name']

            dataset_dicts['fgimg_mask'] = fgimg_mask.astype('uint8')  # [H, W] # 1 for black, 0 for white
            # assert 1 == 0
            if self.use_freq_guild:
                dataset_dicts['low_freq_guild'] = low_freq_guild

            return dataset_dicts

        if self.split == 'test' and self.data_name == 'dsb' and 1==0:
            file_root = dataset_dict['file_root']
            dataset = self.load_data_for_dsb(file_root, data_key)
            img = dataset['image']

            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            if self.use_freq_guild:
                low_freq_guild = self.generate_freq_guild(gray_img)

            fgimg_mask = gray_img < 200  # black: True, white: False

            # normalize image
            img = self.map(img)

            dataset_dicts["width"] = self.width
            dataset_dicts["height"] = self.height

            dataset_dicts['image'] = torch.as_tensor(
                np.ascontiguousarray(img.transpose(2, 0, 1))
            )  # [3, H, W]

            dataset_dicts['file_name'] = dataset['image_name']

            dataset_dicts['fgimg_mask'] = fgimg_mask.astype('uint8')  # [H, W] # 1 for black, 0 for white
            # assert 1 == 0
            if self.use_freq_guild:
                dataset_dicts['low_freq_guild'] = low_freq_guild

            return dataset_dicts


        dataset = self.datagenerate(data_key)
        img = dataset['image']

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.use_freq_guild:
            low_freq_guild = self.generate_freq_guild(gray_img)

        fgimg_mask = gray_img < 200 # black: True, white: False
        #cv2.imshow("img", img)
        #cv2.waitKey(0)
        #cv2.imshow('mask', fgimg_mask.astype('float32'))
        #cv2.waitKey(0)
        #assert 1 == 0

        labels = dataset['labels']

        # generate contours and fourier
        gen = self.gen
        gen.feed(labels=labels)

        labels_ = gen.reduced_labels


        # plt.subplot(121)
        # plt.imshow(img)
        # plt.subplot(122)
        #
        # plt.imshow(labels_)
        # plt.show()
        # plt.close()
        # assert 1 == 0

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # normalize image
        img = self.map(img)

        dataset_dicts["width"] = self.width
        dataset_dicts["height"] = self.height

        dataset_dicts['image'] = torch.as_tensor(
            np.ascontiguousarray(img.transpose(2, 0, 1))
        )  # [3, H, W]
        dataset_dicts['bmask'] = dataset['bmask'] # [H, W]
        dataset_dicts['labels'] = labels_  # [H, W]
        dataset_dicts['fouriers'] = (gen.fourier.astype('float32'), )
        dataset_dicts['locations'] = (gen.locations.astype('float32'), )
        dataset_dicts['sampled_contours'] = (gen.sampled_contours.astype('float32'), )
        dataset_dicts['sampling'] = (gen.sampling.astype('float32'), )
        dataset_dicts['file_name'] = dataset['image_name']

        dataset_dicts['fgimg_mask'] = fgimg_mask.astype('uint8') # [H, W] # 1 for black, 0 for white
        if self.use_freq_guild:
            dataset_dicts['low_freq_guild'] = low_freq_guild.astype('float32')  # [0, 1], float

        # for k in dataset_dicts.keys():
        #     v = dataset_dicts[k]
        #     print(k, end=" ")
        #     if isinstance(v, str):
        #         print(v)
        #     elif isinstance(v, int):
        #         print(v)
        #     else:
        #         print(v.shape)
        # image
        # torch.Size([3, 512, 512])
        # labels
        # torch.Size([512, 512])
        # fouriers
        # torch.Size([3, 25, 4])
        # locations
        # torch.Size([3, 2])
        # sampled_contours
        # torch.Size([3, 128, 2])
        # sampling
        # torch.Size([128])

        # assert 1 == 0
        return dataset_dicts





