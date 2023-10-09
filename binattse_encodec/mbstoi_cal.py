

import warnings
import numpy as np
from scipy.io import wavfile
from clarity.evaluator.mbstoi.mbstoi import mbstoi

import os
import numpy as np
from scipy.io import wavfile

def computeMBSTOI(clean_folder_path, noisy_folder_path):
    clean_files = []
    noisy_files = []

    for filename in os.listdir(clean_folder_path):
        if filename.endswith("target_CH1.wav"):
            clean_files.append(filename)

    for filename in os.listdir(noisy_folder_path):
        if filename.endswith("output.wav"):
            noisy_files.append(filename)
    # print('first 6 clean file', clean_files[0:5])
    # print('first 6 noisy file', noisy_files[0:5])
    total_mbstoi = 0.0
    num_files = len(clean_files)
    num_files_noisy = len(noisy_files)
    # print('clean_files number', num_files)
    # print('noisy number', num_files_noisy)

    for i in range(num_files):
        clean_file = clean_files[i]
        noisy_file = noisy_files[i]

        clean_file_path = os.path.join(clean_folder_path, clean_file)
        noisy_file_path = os.path.join(noisy_folder_path, noisy_file)

        fs_clean, clean_audio = wavfile.read(clean_file_path)
        fs_noisy, noisy_audio = wavfile.read(noisy_file_path)

        clean_audio_left = clean_audio[:, 0]
        clean_audio_right = clean_audio[:, 1]
        noisy_audio_left = noisy_audio[:, 0]
        noisy_audio_right = noisy_audio[:, 1]

        mbstoi_score = mbstoi(clean_audio_left, clean_audio_right, noisy_audio_left, noisy_audio_right, fs_clean)
        print(mbstoi_score)
        total_mbstoi += mbstoi_score
    results_folder_path = "results"
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)
    average_mbstoi = total_mbstoi / num_files
    print("average_mbstoi:", average_mbstoi)

    # np.savetxt("average_mbstoi.txt", [average_mbstoi], fmt='%.6f')
    result_file_path = os.path.join(results_folder_path, "mbstoi.txt")
    np.savetxt(result_file_path, [average_mbstoi], fmt='%.6f')

    return average_mbstoi




if __name__ == "__main__":
    clean_folder_path = "data/test"
    models_folder_path = "models"
    model_subfolders = [folder for folder in os.listdir(models_folder_path) if os.path.isdir(os.path.join(models_folder_path, folder))]

    if model_subfolders:

        first_model_subfolder = model_subfolders[0]
        noisy_folder_path = os.path.join(models_folder_path, first_model_subfolder, "audio")
        average_mbstoi_first = computeMBSTOI(clean_folder_path, noisy_folder_path)


        result_file_path = os.path.join("results", f"mbstoi_{first_model_subfolder}.txt")
        np.savetxt(result_file_path, [average_mbstoi_first], fmt='%.6f')


    noisy_folder_path_fixed = "results/audio"
    average_mbstoi_fixed = computeMBSTOI(clean_folder_path, noisy_folder_path_fixed)

    result_file_path_fixed = os.path.join("results", "mbstoi_final.txt")
    np.savetxt(result_file_path_fixed, [average_mbstoi_fixed], fmt='%.6f')