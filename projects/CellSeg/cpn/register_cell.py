import os.path as osp
import json
import os
import random

from detectron2.data import DatasetCatalog, MetadataCatalog
import pickle
import numpy as np
import scipy
import pickle

def register_cell_cos():
    # excimg = os.listdir("/home/kyfq/detectron2/inferencevis_1712718119.7641475")
    excimg = []

    img_files = "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/new_seg_data/jzx/"
    # img_files = "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/single_test"
    all_files = os.listdir(img_files) # ori_imgs
    cimg_names = []
    for e in all_files:
        if e.split(".")[-1] not in ['png', 'bmp', 'jpg']:
            # print(e)
            continue

        f = False
        for ex in excimg:
            if e in ex:
                f = True
                break
        if f:
            print("#################### already inference img: ", e)
            continue

        cimg_names.append(e)
    # print(cimg_names)
    # assert 1 == 0
    data_dict = []
    for i in range(len(cimg_names)):
        file_name = cimg_names[i]
        data_dict.append({
            'data_index': i,
            'key': file_name,
            'file_root': img_files
        })

    img_files = "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/new_seg_data/tct/"
    all_files = os.listdir(img_files)  # ori_imgs
    cimg_names = []
    for e in all_files:
        if e.split(".")[-1] not in ['png', 'bmp', 'jpg']:
            # print(e)
            continue

        f = False
        for ex in excimg:
            if e in ex:
                f = True
                break
        if f:
            print("#################### already inference img: ", e)
            continue

        cimg_names.append(e)
    for i in range(len(cimg_names)):
        file_name = cimg_names[i]
        data_dict.append({
            'data_index': i,
            'key': file_name,
            'file_root': img_files
        })
    # print(data_dict)
    # assert 1 == 0
    return data_dict

def register_cell(split='train'):
    # assert 1 == 0
    print("注册", split)
    json_file = "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/合并标注数据/data_512_new/annos.json"
    img_files = "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/合并标注数据/data_512_new/img/"
    # all_files = os.listdir(img_files) # ori_imgs
    # with open("/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/合并标注数据/data_512_new/legal_img_files.pkl", 'rb') as f:
    assert split in ('train', 'test')
    with open("/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/合并标注数据/data_512_new/legal_"+split+"_files.pkl", 'rb') as f:
        all_files = pickle.load(f)
    # print(all_files)
    # assert 1 == 0
    data_dict = []

    for i in range(len(all_files)):
        file_name = all_files[i]
        if file_name == 'dfa69dce-d7e7-11ec-a8b3-d562838f25df_2.bmp':
            # print("dddd")
            # assert 1 == 0
            file_name = 'dfa69dce-d7e7-11ec-a8b3-d562838f25df_2.png'
        if file_name in ['TC19008021_13940_8923_1274_1079.png', 'TC19008200_34607_40749_1216_1036.png', 'TC19008200_46727_26557_1266_1225.png', 'TC19009815_10871_27086_1261_1186.png', 'TC19008175_29692_17883_1400_1119.png', 'TC19008234_42265_30573_1349_1180.png', 'TC19009820_34389_18968_1075_1044.png', 'TC19008021_27760_29315_1149_1190.png', 'TC19009815_22369_42838_1412_1093.png', 'TC19009820_16251_19508_1141_1087.png', 'TC19024734_39438_24273_1113_1054.png', 'TC19008141_57171_38701_1399_1143.png', 'TC19008141_17983_30341_1380_1249.png', 'TC19008175_28071_26910_1243_1127.png', 'TC19009820_16992_25644_1083_1057.png', 'TC19008200_19951_25530_1126_1154.png', 'TC19008021_20500_44351_1157_1057.png', 'TC19008021_10482_25873_1170_1069.png', 'TC19024734_8283_31922_1306_1177.png', 'TC19024734_36345_36259_1354_1118.png', 'TC19008021_36749_35668_1205_1030.png', 'TC19024725_5721_32929_1131_1196.png', 'TC19008141_19166_18763_1358_1124.png', 'TC19008200_47878_43024_1322_1256.png', 'TC19008234_46937_54600_1186_1057.png', 'TC19008175_6086_28861_1151_1090.png', 'TC19024725_21896_22785_1396_1059.png', 'TC19024731_51780_36988_1390_1239.png', 'TC19008234_53711_16581_1290_1162.png', 'TC19024734_30497_6433_1197_1055.png', 'TC19008234_24889_7867_1629_1404.png', 'TC19008141_26820_8158_1261_1173.png', 'TC19008021_20024_7067_1152_1152.png', 'TC18006277_20932_27711_1923_1334.png', 'TC19009815_30304_34350_1354_1166.png', 'TC19008141_38066_23649_1308_1169.png', 'TC19024731_25187_55064_1372_1152.png', 'TC19008200_16538_37357_1202_1151.png', 'TC19024725_40491_10961_1157_1229.png', 'TC19009815_28977_11203_1125_1092.png', 'TC19008200_16678_43315_1314_1137.png', 'TC19008141_45298_43435_1161_1070.png', 'TC19024734_10771_13844_1173_1088.png', 'TC19008175_24026_39538_1239_1067.png', 'TC19024734_9202_43289_1235_1122.png', 'TC19008175_16815_17238_1147_1119.png', 'TC19024731_40135_31235_1460_1250.png', 'TC19024725_36159_26968_1162_1228.png', 'TC19009815_39363_17452_1560_1258.png', 'TC19008175_50054_48782_1294_1174.png', 'TC19024731_37633_51682_1444_1434.png', 'TC19024731_28046_31793_1353_1238.png', 'TC19008175_11941_39796_1203_1083.png', 'TC19024734_11212_9806_1266_1078.png', 'TC19008021_17908_25809_1174_1028.png', 'TC19008141_37378_52308_1103_1050.png', 'TC19009820_28398_34115_1160_1042.png', 'TC19009815_51635_9005_1297_1125.png', 'TC19008200_31917_44872_1187_1041.png', 'TC19009820_30652_14279_1316_1172.png', 'TC19008234_16030_18864_1352_1263.png', 'TC19024725_41436_38136_1533_1199.png', 'TC19009820_43853_32345_1204_1135.png', 'TC19009820_36095_25223_1141_1125.png', 'TC19024725_10047_22197_1166_1069.png', 'TC19024731_63309_31387_1469_1315.png', 'TC19008200_48943_19828_1309_1053.png', 'TC19024734_34859_4520_1193_1097.png', 'TC19024725_21508_14475_1755_1173.png', 'TC19024734_47768_19753_1339_1148.png', 'TC19008175_34562_45789_1172_1049.png', 'TC19024725_22810_31843_1459_1221.png', 'TC19008234_25319_53082_1657_1325.png', 'TC19008021_11041_30352_1099_1195.png', 'TC19009815_47063_34647_1278_1103.png', 'TC19009815_14070_32383_1387_1225.png', 'TC19009815_37420_40492_1294_1198.png', 'TC19008200_53501_34385_1177_1180.png', 'TC19009820_30831_43563_1189_1026.png', 'TC19024725_39381_20163_1409_1166.png', 'TC19009820_17509_32322_1055_1036.png', 'TC19008021_20032_37784_1224_1047.png', 'TC19008021_26744_23401_1031_1054.png', 'TC19024731_31917_56081_1340_1250.png', 'TC19008141_27461_39241_1372_1230.png', 'TC19009820_27522_20659_1402_1203.png', 'TC19008234_27647_2871_1806_1583.png', 'TC19009815_18627_20217_1355_1363.png', 'TC19024731_14726_29897_1451_1462.png', 'TC19008175_47750_33437_1267_1103.png', 'TC19008200_26115_9846_1151_1031.png', 'TC19024731_54117_45548_1324_1321.png', 'TC19008234_11883_36123_1414_1267.png', 'TC19008141_25227_49724_1197_1097.png', 'TC19024734_20562_30199_1218_1120.png', 'TC19008175_46710_13160_1278_1086.png', 'TC19008141_46207_21427_1272_1139.png']:
            continue
        data_dict.append({
            'data_index': i,
            'key': file_name,
        })
        # if len(data_dict) >= 4500: # for ori images, 5700, 4500 for train
        # if len(data_dict) > 3500:  # for rimages, 4230, 350 for train
        #     break

    if split == 'train':
        random.shuffle(data_dict)
    # if len(data_dict) == 0:
    #     data_dict.append({
    #         'image_id': 142,
    #         'ref_id': 0,
    #         'raw_sent': ""
    #     })

    print("data for " + split + ": ", len(data_dict))
    # assert 1 == 0
    return data_dict


def register_cell_all():
    print("注册cell数据集")

    split = 'train'
    split = 'test'
    # split = 'cos'

    if split == 'train':

        DatasetCatalog.register("cell_" + "train", lambda: register_cell())
        MetadataCatalog.get('cell_' + "train").set(json_file=None, evaluator_type="refcoco", thing_classes=["cell"])
    
    elif split == 'test':
    # split = 'test'

        DatasetCatalog.register("cell_" + 'test', lambda: register_cell(split=split))
        MetadataCatalog.get('cell_' + 'test').set(json_file=None, evaluator_type="refcoco", thing_classes=["cell"])

    elif split == 'cos':
        DatasetCatalog.register("cell_" + 'test', lambda: register_cell_cos())
        MetadataCatalog.get('cell_' + 'test').set(json_file=None, evaluator_type="refcoco", thing_classes=["cell"])

    else:
        raise()

register_cell_all()