import numpy as np
import cv2
from skimage import io
from osgeo import gdal
import os
import math

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ListAverageMeter(object):
    """Computes and stores the average and current values of a list"""
    def __init__(self):
        self.len = 10000  # set up the maximum length
        self.reset()

    def reset(self):
        self.val = [0] * self.len
        self.avg = [0] * self.len
        self.sum = [0] * self.len
        self.count = 0

    def set_len(self, n):
        self.len = n
        self.reset()

    def update(self, vals, n=1):
        assert len(vals) == self.len, 'length of vals not equal to self.len'
        self.val = vals
        for i in range(self.len):
            self.sum[i] += self.val[i] * n
        self.count += n
        for i in range(self.len):
            self.avg[i] = self.sum[i] / self.count
            

def cv2_read_img(filename):
    img = cv2.imread(filename,-1)
    
    if img.dtype == 'uint8':
        return img[:, :, ::-1].astype('float32') / 255.0
    else:
        return img[:, :, ::-1]


def sk_read_img(filename):
    
    img = io.imread(filename)

    if img.dtype == 'uint8':
        return img.astype('float32') / 255.0
    elif img.dtype == 'uint16':
        return img.astype('float32') / 10000.0
    else:
        return img


def RGBwrite_img(filename, img):
    img = np.round((img[:, :, ::-1].copy() * 255.0)).astype('uint8')
    cv2.imwrite(filename, img)


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1]).copy()


def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0]).copy()


def gdal_read_img(filename):
    dataset = gdal.Open(filename)
    # 图像的列数（宽度）
    im_width = dataset.RasterXSize
    # 图像的行数（高度）
    im_height = dataset.RasterYSize
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    del dataset
    return im_proj, im_geotrans, im_width, im_height, im_data


def gdal_write_img(filename, im_proj, im_geotrans, im_data):
    # im_data 数据格式为（bands,w,h）
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    # 栅格数据int型，则改为gdal.GDT_Int32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
    if im_geotrans !=None :
        dataset.SetGeoTransform(im_geotrans)
    if im_proj != None:
        dataset.SetProjection(im_proj)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    dataset.FlushCache()
    dataset = None
    del dataset


def getFileName(path, expname):
    ''' 获取指定目录下的所有指定后缀的文件名 '''
    # usage： n = getFileName（目录，扩展名） return： 指定后缀名的列表
    f_list = os.listdir(path)
    out_list = []
    cnt = 0

    # print f_list
    for i in f_list:
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] == expname:
            out_list.append(i)
            cnt += 1
    return out_list


def get_filelist(dir):
    '''获取目录下，包含所有子文件夹下的，所有文件名（不包含文件夹名）'''
    Filelist = []
    fnamelist = []
    for home, dirs, files in os.walk(dir):
        for filename in files:
            # 文件名列表，包含完整路径
            Filelist.append(os.path.join(home, filename))
            # # 文件名列表，只包含文件名
            fnamelist.append(filename)
    return Filelist, fnamelist


def rmse2np(y_true, y_pred):
    """
    RMSE for regression to process numpy's array type
    :param y_true: array tensor for observation, just the last output .
    :param y_pred: array tensor for predictions, just the last output.
    :return: rmse
    """
    error = y_true-y_pred
    ret = np.sqrt(np.mean(np.square(error)))
    return ret


def imgdataRegularization(indata, mean_=None, std_=None):
    '''

    :param indata:输入矩阵数据
    :return:返回 标准化后的数据
    '''
    if mean_ == None:
        if len(indata.shape) == 2:
            rows, colums = indata.shape
            mean = np.mean(indata)
            std = np.std(indata)
            indata = (indata - mean) / std

        if len(indata.shape) == 3:
            n_band, rows, colums = indata.shape
            for i in range(n_band):
                data = indata[n_band, :, :]
                mean = np.mean(data)
                std = np.std(data)
                data = (data - mean) / std
                indata[n_band, :, :] = data
    if mean_ != None:
        if len(indata.shape) == 2:
            indata = (indata - mean_) / std_

        if len(indata.shape) == 3:
            n_band, rows, colums = indata.shape
            for i in range(n_band):
                indata[n_band, :, :] = (indata[n_band, :, :] - mean_) / std_
    return indata


def computeCorrelation(X, Y):
    '''

    :param X:
    :param Y:
    :return: 计算相关性r和r平方
    '''
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2

    SST = math.sqrt(varX * varY)
    print("r：", SSR / SST, "r-squared：", (SSR / SST) ** 2)
    return


def read_sentinel_2(xml_filename, resolution=10):
    '''

    :param xml_filename: 哨兵2.save文件夹里MTD_MSIL1C.xml文件
    :param resolution: 需要读取的分辨率数据
    :return: 返回图像信息
    '''
    imgfile = gdal.Open(xml_filename)
    ds_list = imgfile.GetSubDatasets()
    if resolution == 10:
        img_ds = gdal.Open(ds_list[0][0])
        img_arr = img_ds.ReadAsArray()
        img_arr = img_arr[[2, 1, 0, 3], :, :]  # 原始波段顺序为rgbnir,转为bgrnir
    elif resolution == 20:
        img_ds = gdal.Open(ds_list[1][0])
        img_arr = img_ds.ReadAsArray()  # 原始波段顺序为B5,6,7,8A,11,12
    elif resolution == 60:
        img_ds = gdal.Open(ds_list[2][0])
        img_arr = img_ds.ReadAsArray()  # 原始波段顺序为B1,B9,B10

    img_proj = img_ds.GetProjection()
    img_geotrans = img_ds.GetGeoTransform()
    img_res = abs(img_geotrans[1])
    img_width = img_ds.RasterXSize
    img_height = img_ds.RasterYSize

    return img_proj, img_geotrans, img_width, img_height, img_arr


def ReadS2Bands(input, data_level='L1C', resolution=10):
    # 读取13波段的哨兵2数据，并采样至指定分辨率
    if resolution == 10:
        out_xsize, out_ysize = 10980, 10980
    elif resolution == 20:
        out_xsize, out_ysize = 5490, 5490
    elif resolution == 60:
        out_xsize, out_ysize = 1830, 1830

    imgfile = gdal.Open(input)

    ds_list = imgfile.GetSubDatasets()  # 获取子数据集。该数据以数据集形式存储且以子数据集形式组织
    # 打开第1个数据子集的路径。ds_list有4个子集，内部前段是路径，后段是数据信息
    L1C_img_ds = gdal.Open(ds_list[0][0])
    img_10m_proj = L1C_img_ds.GetProjection()
    img_10m_geotrans = L1C_img_ds.GetGeoTransform()
    img_xsize = L1C_img_ds.RasterXSize
    img_ysize = L1C_img_ds.RasterYSize
    L1C_img_10_arr = L1C_img_ds.ReadAsArray()  # 将数据集中的数据读取为ndarray
    # 原始波段顺序为rgbnir,转为bgrnir --  b2,b3,b4,b8
    L1C_img_10_arr = L1C_img_10_arr[[2, 1, 0, 3], :, :]
    if img_xsize == out_xsize:
        pass
    else:
        L1C_img_10_arr = np.transpose(L1C_img_10_arr, [1, 2, 0])
        L1C_img_10_arr = cv2.resize(
            L1C_img_10_arr, (out_xsize, out_ysize), interpolation=cv2.INTER_LINEAR)
        L1C_img_10_arr = np.transpose(L1C_img_10_arr, [2, 0, 1])

    # 读取20m分辨率波段, b5,b6,b7,b8a,b11,b12
    l1c_img_20_ds = gdal.Open(ds_list[1][0])
    img_20m_proj = l1c_img_20_ds.GetProjection()
    img_20m_geotrans = l1c_img_20_ds.GetGeoTransform()
    img_xsize = l1c_img_20_ds.RasterXSize
    img_ysize = l1c_img_20_ds.RasterYSize
    l1c_img_20_arr = l1c_img_20_ds.ReadAsArray()[0:6, :, :]
    if img_xsize == out_xsize:
        pass
    else:
        l1c_img_20_arr = np.transpose(l1c_img_20_arr, [1, 2, 0])
        l1c_img_20_arr = cv2.resize(
            l1c_img_20_arr, (out_xsize, out_ysize), interpolation=cv2.INTER_LINEAR)  # 重采样
        l1c_img_20_arr = np.transpose(l1c_img_20_arr, [2, 0, 1])

    # 读取60m分辨率波段，b1,b9,b10
    l1c_img_60_ds = gdal.Open(ds_list[2][0])
    img_60m_proj = l1c_img_60_ds.GetProjection()
    img_60m_geotrans = l1c_img_60_ds.GetGeoTransform()
    img_xsize = l1c_img_60_ds.RasterXSize
    img_ysize = l1c_img_60_ds.RasterYSize
    if data_level == 'L1C':
        l1c_img_60_arr = l1c_img_60_ds.ReadAsArray()[[0, 1, 2], :, :]
    else:
        # L2A 数据没有band10
        l1c_img_60_arr = l1c_img_60_ds.ReadAsArray()[[0, 1], :, :]
    if img_xsize == out_xsize:
        pass
    else:
        l1c_img_60_arr = np.transpose(l1c_img_60_arr, [1, 2, 0])
        l1c_img_60_arr = cv2.resize(
            l1c_img_60_arr, (out_xsize, out_ysize), interpolation=cv2.INTER_LINEAR)  # 重采样到10m分辨率
        l1c_img_60_arr = np.transpose(l1c_img_60_arr, [2, 0, 1])

    # 13个波段叠加，顺序为b1,b2,b3,b4,b5,b6,b7,b8,b8a,b9,b10,b11,b12
    cat_img_arr = np.concatenate((np.expand_dims(l1c_img_60_arr[0, :, :], axis=0), L1C_img_10_arr[0:3, :, :], l1c_img_20_arr[0:4, :, :],
                                  np.expand_dims(L1C_img_10_arr[3, :, :], axis=0), l1c_img_60_arr[1:, :, :],  l1c_img_20_arr[4:, :, :]), axis=0)
    
    if resolution == 10:
        img_proj, img_geotrans = img_10m_proj, img_10m_geotrans
    elif resolution == 20:
        img_proj, img_geotrans = img_20m_proj, img_20m_geotrans
    elif resolution == 60:
        img_proj, img_geotrans = img_60m_proj, img_60m_geotrans

    return img_proj, img_geotrans, out_xsize, out_ysize, cat_img_arr

if __name__ == '__main__':
    # img_proj, img_geotrans, img_xsize, img_ysize, cat_img_arr = ReadS2Bands(r"G:\DataSets\OSCR-THIN\train\clear\S2A_MSIL1C_20220323T080611_N0400_R078_T35KKP_20220323T100758.SAFE\MTD_MSIL1C.xml",resolution=20)
    a = sk_read_img(r"G:\DataSets\OSCR-THIN\train_clip_20m\haze_thumb\T10TDQ_20220815T223626_000_008_6_5_8_1_0.16.tif")
    b = sk_read_img(r"G:\DataSets\OSCR-THIN\train_clip_20m\clear\T10TDQ_20220815T223626_000_008.tif")

    print(1)

        