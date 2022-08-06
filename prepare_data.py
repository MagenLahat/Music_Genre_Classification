import numpy as np
import librosa
import os
import json

DATASET_PATH = "dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050  # 1 sec. of audio


def splitsongs(X, overlap=0.5, chunk=22050):
    # Empty lists to hold our results
    temp_X = []

    # Get the input song array size
    xshape = X.shape[0]
    # chunk = int(xshape*window)
    offset = int(chunk * (1. - overlap))
    # Split the song and create new ones on windows
    spsong = [X[i:i + chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        if s.shape[0] != chunk:
            continue

        temp_X.append(s)

    return np.array(temp_X)


def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):
    # dictionary where we'll store mapping, labels, MFCCs and filenames
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're at sub-folder level
        if dirpath is not dataset_path:

            # save label (i.e., sub-folder name) in the mapping
            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            # process all audio files in sub-dir and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency among different files
                signal, sample_rate = librosa.load(file_path)

                splits = splitsongs(signal)
                for split in splits:

                    # drop audio files with less than pre-decided number of samples
                    if len(split) >= SAMPLES_TO_CONSIDER:
                        # ensure consistency of the length of the signal
                        split = split[:SAMPLES_TO_CONSIDER]

                        # extract MFCCs
                        MFCCs = librosa.feature.mfcc(split, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                     hop_length=hop_length)

                        # store data for analysed track
                        data["MFCCs"].append(MFCCs.T.tolist())
                        data["labels"].append(i - 1)
                        data["files"].append(file_path)
                        print("{}: {}".format(file_path, i - 1))

    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)
