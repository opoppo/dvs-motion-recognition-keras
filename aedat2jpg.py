"""
Function:
1. Transform aedat file into jpg with a designated coding method including frequency, sae, snn, mtc.
2. Generate csv file.
"""

import os
import struct
import cv2
import numpy as np
import pandas as pd
import random


def getDVSeventsDavis(file, ROI=np.array([]), numEvents=1e10, startEvent=0, startTime=0):
    print('\ngetDVSeventsDavis function called \n')
    sizeX = 346
    sizeY = 260
    x0 = 0
    y0 = 0
    x1 = sizeX
    y1 = sizeY
    if len(ROI) != 0:
        if len(ROI) == 4:
            print('Region of interest specified')
            x0 = ROI(0)
            y0 = ROI(1)
            x1 = ROI(2)
            y1 = ROI(3)
        else:
            print(
                'Unknown ROI argument. Call function as: \n getDVSeventsDavis(file, ROI=[x0, y0, x1, y1], numEvents=nE, startEvent=sE) to specify ROI or\n getDVSeventsDavis(file, numEvents=nE, startEvent=sE) to not specify ROI')
            return

    else:
        print('No region of interest specified, reading in entire spatial area of sensor')

    print('Reading in at most', str(numEvents))
    print('Starting reading from event', str(startEvent))

    triggerevent = int('400', 16)
    polmask = int('800', 16)
    xmask = int('003FF000', 16)
    ymask = int('7FC00000', 16)
    typemask = int('80000000', 16)
    typedvs = int('00', 16)
    xshift = 12
    yshift = 22
    polshift = 11
    x = []
    y = []
    ts = []
    pol = []
    numeventsread = 0

    length = 0
    aerdatafh = open(file, 'rb')
    k = 0
    p = 0
    statinfo = os.stat(file)
    if length == 0:
        length = statinfo.st_size
    print("file size", length)

    lt = aerdatafh.readline()
    while lt and str(lt)[2] == "#":
        p += len(lt)
        k += 1
        lt = aerdatafh.readline()
        continue

    aerdatafh.seek(p)
    tmp = aerdatafh.read(8)
    p += 8
    while p < length:
        ad, tm = struct.unpack_from('>II', tmp)
        ad = abs(ad)
        if tm >= startTime:
            if (ad & typemask) == typedvs:
                xo = sizeX - 1 - float((ad & xmask) >> xshift)
                yo = float((ad & ymask) >> yshift)
                polo = 1 - float((ad & polmask) >> polshift)
                if xo >= x0 and xo < x1 and yo >= y0 and yo < y1:
                    x.append(xo)
                    y.append(yo)
                    pol.append(polo)
                    ts.append(tm)
        aerdatafh.seek(p)
        tmp = aerdatafh.read(8)
        p += 8
        numeventsread += 1

    print('Total number of events read =', numeventsread)
    print('Total number of DVS events returned =', len(ts))

    # ts[:] = [x - ts[0] for x in ts]  # absolute time -> relative time
    # x[:] = [int(a) for a in x]
    # y[:] = [int(a) for a in y]

    return ts, x, y, pol


# Transform aedat to jpg frames using decoding method frequency with designated step time in millisecond.
# Return count of frames.
def frequency(input_file, outputpath, step_time_ms=50,slice_time_ms=10, white_threshold=500000):
    assert step_time_ms > slice_time_ms
    T, X, Y, Pol = getDVSeventsDavis(input_file)
    T = np.array(T).reshape((-1, 1))
    X = np.array(X).reshape((-1, 1))
    Y = np.array(Y).reshape((-1, 1))
    Pol = np.array(Pol).reshape((-1, 1))
    step_time = step_time_ms * 1000
    slice_time = slice_time_ms * 1000
    start_idx = 0
    end_idx = 0
    slice_idx = 0
    start_time = T[0]
    end_time = start_time + step_time
    slice_end_time = start_time + slice_time
    img_count = 1

    while end_time <= T[-1]:

        while T[slice_idx] < slice_end_time:
            slice_idx += 1

        while T[end_idx] < end_time:
            end_idx = end_idx + 1
        data_x = np.array(X[start_idx:end_idx]).reshape((-1, 1))
        data_y = np.array(Y[start_idx:end_idx]).reshape((-1, 1))
        data = np.column_stack((data_x, data_y)).astype(np.int32)
        counter = np.zeros((260, 346))

        for i in range(0, data.shape[0]):
            counter[data[i, 1], data[i, 0]] += 1
        img = np.flip(255 * 2 * (1 / (1 + np.exp(-counter)) - 0.5), 0)
        output_dir = outputpath
        output_file = input_file.split('/')[-1].split('.')[0]
        # img_name = os.path.join(output_dir, "try/{0}-{1:04d}.jpg".format(output_file, img_count))
        img_name = outputpath + "_%03d.jpg" % img_count
        start_time += slice_time
        slice_end_time = start_time + slice_time
        end_time = start_time + step_time
        start_idx = slice_idx

        if sum(img[img > 0]) > white_threshold:
            cv2.imwrite(img_name, img)
            img_count += 1

    return img_count - 1


# Transform aedat to jpg frames using decoding method SAE with designated step time in millisecond.
# Return count of frames.
def sae(input_file, outputpath, step_time_ms=50,slice_time_ms=10,  white_threshold=500000):
    assert step_time_ms > slice_time_ms
    T, X, Y, Pol = getDVSeventsDavis(input_file)
    T = np.array(T).reshape((-1, 1))
    X = np.array(X).reshape((-1, 1))
    Y = np.array(Y).reshape((-1, 1))
    Pol = np.array(Pol).reshape((-1, 1))
    step_time = step_time_ms * 1000
    slice_time = slice_time_ms * 1000
    start_idx = 0
    end_idx = 0
    slice_idx = 0
    start_time = T[0]
    end_time = start_time + step_time
    slice_end_time = start_time + slice_time
    img_count = 1

    while end_time <= T[-1]:
        # end_time = start_time + step_time
        while T[slice_idx] < slice_end_time:
            slice_idx += 1

        while T[end_idx] < end_time:
            end_idx = end_idx + 1

        data_x = np.array(X[start_idx:end_idx]).reshape((-1, 1))
        data_y = np.array(Y[start_idx:end_idx]).reshape((-1, 1))
        data_T = np.array(T[start_idx:end_idx]).reshape((-1, 1))
        data = np.column_stack((data_x, data_y)).astype(np.int32)

        timestamp = start_time * np.ones((260, 346))

        for i in range(0, data.shape[0]):
            timestamp[data[i, 1], data[i, 0]] = data_T[i]
        img = np.flip(255 * (timestamp - start_time) / step_time, 0).astype(np.uint8)
        output_dir = outputpath
        output_file = input_file.split('/')[-1].split('.')[0]
        # img_name = os.path.join(output_dir, "try/{0}-{1:04d}.jpg".format(output_file, img_count))
        img_name = outputpath + "_%03d.jpg" % img_count

        start_time += slice_time
        slice_end_time = start_time + slice_time
        end_time = start_time + step_time
        start_idx = slice_idx

        if sum(img[img > 0]) > white_threshold:
            cv2.imwrite(img_name, img)
            img_count += 1

    return img_count - 1


# Transform aedat to jpg frames using decoding method LIF with designated step time in millisecond.
# Return count of frames.
def lif(input_file, outputpath, step_time_ms=50,slice_time_ms=10, white_threshold=500000):
    T, X, Y, Pol = getDVSeventsDavis(input_file)
    T = np.array(T).reshape((-1, 1))
    X = np.array(X).reshape((-1, 1))
    Y = np.array(Y).reshape((-1, 1))
    Pol = np.array(Pol).reshape((-1, 1))
    step_time = step_time_ms * 1000
    slice_time = slice_time_ms * 1000
    start_idx = 0
    end_idx = 0
    slice_idx = 0
    start_time = T[0]
    end_time = start_time + step_time
    slice_end_time = start_time + slice_time
    img_count = 1

    while end_time <= T[-1]:

        while T[slice_idx] < slice_end_time:
            slice_idx += 1

        while T[end_idx] < end_time:
            end_idx = end_idx + 1
        data_x = np.array(X[start_idx:end_idx]).reshape((-1, 1))
        data_y = np.array(Y[start_idx:end_idx]).reshape((-1, 1))
        data = np.column_stack((data_x, data_y)).astype(np.int32)
        counter = np.zeros((260, 346))

        for i in range(0, data.shape[0]):
            counter[data[i, 1], data[i, 0]] += 1
        img = np.flip(((255 * (1 / (1 + (np.exp(-counter) / 2.0)))) - 127) * 2, 0)
        output_dir = outputpath
        output_file = input_file.split('/')[-1].split('.')[0]
        # img_name = os.path.join(output_dir, "try/{0}-{1:04d}.jpg".format(output_file, img_count))
        img_name = outputpath + "_%03d.jpg" % img_count
        start_time += slice_time
        slice_end_time = start_time + slice_time
        end_time = start_time + step_time
        start_idx = slice_idx

        if sum(img[img > 0]) > white_threshold:
            cv2.imwrite(img_name, img)
            img_count += 1

    return img_count - 1


if __name__ == "__main__":

    data_file = []
    motion_cat = ['left', 'leftdown', 'down', 'rightdown', 'right']
    motion_num = motion_cat.__len__()
    train_test_split_ratio = 0.8
    imgoutputpath = []
    encoding_cat = ['frequency', 'sae', 'lif', 'fusion']
    encodingID = 0
    train_test_cat = ['test', 'train']

    for motionID in range(motion_num):
        aedatpath = 'I:/DAVIS/headpose/%d' % (motionID + 1) + motion_cat[motionID] + '/'
        imgtrainpath = './data/train/'
        imgtestpath = './data/test/'
        trainsequenceID = [0, 0, 0, 0, 0]  # every subject every motion
        testsequenceID = [0, 0, 0, 0, 0]
        sequenceID = {'train': trainsequenceID, 'test': testsequenceID}

        for fpathe, dirs, fs in os.walk(aedatpath):
            random.shuffle(fs)
            # fs=fs[:5]     # for quick test!
            for pp, f in enumerate(fs):
                frame_num = 0
                train_test_ind = -1  # 1 for train; 0 for test

                if pp < len(fs) * train_test_split_ratio:
                    imgoutputpath = imgtrainpath + motion_cat[motionID] + '/'
                    train_test_ind = 1
                    trainsequenceID[motionID] += 1
                else:
                    imgoutputpath = imgtestpath + motion_cat[motionID] + '/'
                    train_test_ind = 0
                    testsequenceID[motionID] += 1

                file = aedatpath + f

                if not os.path.exists(imgoutputpath):
                    os.makedirs(imgoutputpath)

                if train_test_ind == 1 or train_test_ind == 0:
                    imgFullFile = imgoutputpath + ('v_%s_g%02d_c%02d' % (
                        motion_cat[motionID], 0, sequenceID[train_test_cat[train_test_ind]][motionID]))

                    if encoding_cat[encodingID] == 'frequency':
                        frame_num = lif(file, imgFullFile)

                    data_file.append([train_test_cat[train_test_ind], motion_cat[motionID],
                                      'v_%s_g%02d_c%02d' % (
                                          motion_cat[motionID], 0,
                                          sequenceID[train_test_cat[train_test_ind]][motionID]),
                                      frame_num])

    df = pd.DataFrame(data_file)
    df.to_csv('./data/data_file.csv', index=0, index_label=0)
