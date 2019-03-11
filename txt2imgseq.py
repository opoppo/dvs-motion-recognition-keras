import numpy as np
import cv2
import os
import pandas as pd


def loadTxt(file):
    count = 0
    ts = []
    x = []
    y = []
    pol = []
    with open(file) as f:
        for line in f:
            items = line.split()
            ts.insert(count, float(items[0]))
            x.insert(count, int(items[1]))
            y.insert(count, int(items[2]))
            pol.insert(count, int(items[3]))

            count = count + 1

    return ts, x, y, pol


if __name__ == "__main__":

    data_file = []

    for subjectID in range(1):
        txtpath = 'I:/celexDVS_headpose_data/txt/%d' % subjectID + '/cortxt/'
        imgpath = './data/train/'
        sequenceID = [0, 0, 0]
        for fpathe, dirs, fs in os.walk(txtpath):
            for pp, f in enumerate(fs):
                frame_count = 0
                classID = -1
                if pp % 3 == 0:
                    imgoutputpath = imgpath + '/00/'  # updown
                    classID = 0
                    sequenceID[classID] += 1
                elif (pp - 1) % 3 == 0:
                    imgoutputpath = imgpath + '/01/'  # left
                    classID = 1
                    sequenceID[classID] += 1
                elif (pp - 2) % 3 == 0:
                    imgoutputpath = imgpath + '/02/'  # right
                    classID = 2
                    sequenceID[classID] += 1

                file = txtpath + f

                if not os.path.exists(imgoutputpath):
                    os.makedirs(imgoutputpath)

                t, x, y, pol = loadTxt(file)
                x[:] = [int(a - 1) for a in x]
                y[:] = [int(a - 1) for a in y]

                img = np.zeros((640, 768, 3), dtype=np.uint8)

                idx = 0
                start_idx = 0
                startTime = 0
                endTime = 0
                stepTime = 10000 / 0.08
                frameID = 1

                while startTime < t[-1]:
                    endTime = startTime + stepTime
                    while t[idx] < endTime and idx < len(t) - 1:
                        idx = idx + 1

                    data_x = np.array(x[start_idx:idx]).reshape((-1, 1))
                    data_y = np.array(y[start_idx:idx]).reshape((-1, 1))
                    data_t = np.array(t[start_idx:idx]).reshape((-1, 1))
                    data = np.column_stack((data_x, data_y, data_t))
                    data_filter = data

                    for ii in range(0, data_filter.shape[0]):
                        img[int(data_filter[ii][1] - 1)][int(data_filter[ii][0] - 1)][
                            1] = 255  # channel NONE
                        img[int(data_filter[ii][1] - 1)][int(data_filter[ii][0] - 1)][
                            0] += 85  # channel frequency
                        img[int(data_filter[ii][1] - 1)][int(data_filter[ii][0] - 1)][2] = 255 * (
                                data_filter[ii][2] - t[start_idx]) / (t[idx] - t[
                            start_idx])  # channel time stamp

                    start_idx = idx
                    startTime = t[idx]
                    print(sum(img[img > 0]))
                    if sum(img[img > 0]) > 1000000:
                        cv2.imshow('dvs', img)
                        cv2.waitKey(5)
                        imgFullFile = imgoutputpath + (
                                'v_%02d_g%02d_c%02d_%03d' % (
                            classID, int(subjectID), sequenceID[classID], frameID)) + '.jpg'
                        cv2.imwrite(imgFullFile, img)
                        frameID = frameID + 1

                    img[:] = 0
                    # print('.')
                frame_count = frameID - 1
                data_file.append(
                    ['train', '%02d' % classID, 'v_%02d_g%02d_c%02d' % (classID, subjectID, sequenceID[classID]),
                     frameID])
    df = pd.DataFrame(data_file)
    df.to_csv('./data/data_file.csv', index=0, index_label=0)
