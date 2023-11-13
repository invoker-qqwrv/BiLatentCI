import time
import os
import numpy as np
from scipy.io import wavfile
from clarity.evaluator.mbstoi.mbstoi import mbstoi
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, title, grid, savefig, show, figure

def computeMBSTOI(clean_folder_path, noisy_folder_path):
    clean_files = []
    noisy_files = []

    for filename in os.listdir(clean_folder_path):
        if filename.endswith("target_CH1.wav"):
            clean_files.append(filename)

    for filename in os.listdir(noisy_folder_path):
        if filename.endswith("output.wav"):
            noisy_files.append(filename)

    clean_files = sorted(clean_files, key=extract_numerical_part)
    noisy_files = sorted(noisy_files, key=extract_numerical_part)

    total_mbstoi = 0.0
    num_files = len(clean_files)

    mbstoi_scores = []

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
        mbstoi_scores.append(mbstoi_score)

    average_mbstoi = total_mbstoi / num_files
    print("average_mbstoi:", average_mbstoi)

    np.savetxt("average_mbstoi.txt", [average_mbstoi], fmt='%.6f')

    return np.array(mbstoi_scores)

def extract_numerical_part(s):
    return int(''.join(filter(str.isdigit, s)))

clean_folder_path = "data/test"
models_folder_path = "models"  # Assuming "models" is the parent folder of the first subfolders


model_subfolders = [folder for folder in os.listdir(models_folder_path) if os.path.isdir(os.path.join(models_folder_path, folder))]

mbstoi_data = {}  

if model_subfolders:
    first_model_subfolder = model_subfolders[0]
    audio_bit_folder_path = os.path.join(models_folder_path, first_model_subfolder)

    for subfolder in os.listdir(audio_bit_folder_path):
        if subfolder.startswith("audio_bit="):
            audio_bit_subfolder_path = os.path.join(audio_bit_folder_path, subfolder)
            mbstoi_scores = computeMBSTOI(clean_folder_path, audio_bit_subfolder_path)
            bit = int(subfolder.split("=")[-1])  
            mbstoi_data[bit] = np.mean(mbstoi_scores)  # Store the mean MBSTOI score


# plt.figure(figsize=(8, 6))
plt.plot(list(mbstoi_data.keys()), list(mbstoi_data.values()), marker='o')
plt.xlabel("Audio Bit")
plt.ylabel("Average MBSTOI Score")
plt.title("MBSTOI Scores vs. Audio Bit")
plt.grid(True)



timestamp = time.strftime("%Y%m%d_%H%M%S")
output_pdf_path = f"results/mbstoi_scores_{timestamp}.pdf"

plt.savefig(output_pdf_path, format='pdf')
plt.show()
