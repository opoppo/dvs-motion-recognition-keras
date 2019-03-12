"""

You can change you sequence length and limit to a set number of classes
below.

class_limit is an integer that denotes the first N classes you want to
extract features from. This is useful is you don't want to wait to
extract all 101 classes. For instance, set class_limit = 8 to just
extract features for the first 8 (alphabetical) classes in the dataset.
Then set the same number when training models.


the organization of the frames folder
    -./
        - train
            - class_id
                - v_01_g01_c01_001.jpg (v_classId_subjectId_frameId.jpg)
                - ...
                - v_01_g02...
            - ...
        - test
            - ...
                - ...

There is a need to create a data_file.csv in the root folder:
        - generate data_file.csv
        data_file.append([train_or_test, classname, filename_no_ext, nb_frames])
        z.B. [train|test, class_id, video_id, frames_num]
                [ train, c01, v01, 30]

        - pls refer to the example in ./data

The output feature sequences are stored in
        - ./data/sequences
"""
import numpy as np
import os.path
from data import DataSet
from extractor import Extractor
from tqdm import tqdm

# Set defaults.
seq_length = 40
class_limit = None  # Number of classes to extract. Can be 1-101 or None for all.

# Get the dataset.
data = DataSet(seq_length=seq_length, class_limit=class_limit)

# get the model.
model = Extractor()

# Loop through data. Process bar
pbar = tqdm(total=len(data.data))
for video in data.data:

    # Get the path to the sequence for this video.
    path = os.path.join('data', 'sequences', video[2] + '-' + str(seq_length) + \
        '-features')  # numpy will auto-append .npy

    # Check if we already have it.
    if os.path.isfile(path + '.npy'):
        pbar.update(1)
        continue

    # Get the frames for this video.
    frames = data.get_frames_for_sample(video)

    # Now downsample to just the ones we need.
    # z.B. if we have 80 frames of a video and a seq_length of 40, we need to skip one frame every 2 consequent frames
    # the stride(skip) is (video_length / seq_length)
    frames = data.rescale_list(frames, seq_length)

    # Now loop through and extract features to build the sequence.
    sequence = []
    for image in frames:
        features = model.extract(image)
        sequence.append(features)

    # Save the sequence. features from several frames with skip of a video
    np.save(path, sequence)

    pbar.update(1)

pbar.close()
