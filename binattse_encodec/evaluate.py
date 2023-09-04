import csv
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import hydra
import numpy as np
from numpy import ndarray
from omegaconf import DictConfig
from scipy.signal import unit_impulse
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from clarity.evaluator.mbstoi.mbstoi import mbstoi
from clarity.evaluator.mbstoi.mbstoi_utils import find_delay_impulse
from clarity.evaluator.msbg.msbg import Ear
from clarity.evaluator.msbg.msbg_utils import MSBG_FS, pad
from clarity.utils.audiogram import AUDIOGRAM_REF_CLARITY, Listener
from clarity.utils.file_io import read_signal, write_signal


def listen(ear, signal: ndarray, listener: Listener):
    """
    Generate MSBG processed signal
    :param ear: MSBG ear
    :param wav: binaural signal
    :return: binaural signal
    """
    ear.set_audiogram(listener.audiogram_left)
    out_l = ear.process(signal[:, 0])

    ear.set_audiogram(listener.audiogram_right)
    out_r = ear.process(signal[:, 1])

    if len(out_l[0]) != len(out_r[0]):
        diff = len(out_l[0]) - len(out_r[0])
        if diff > 0:
            out_r[0] = np.flipud(pad(np.flipud(out_r[0]), len(out_l[0])))
        else:
            out_l[0] = np.flipud(pad(np.flipud(out_l[0]), len(out_r[0])))
    return np.concatenate([out_l, out_r]).T


@hydra.main(config_path=".", config_name="config")
def run_HL_processing(cfg: DictConfig) -> None:
    output_path = Path(cfg.path.exp_folder) / "eval_signals"
    output_path.mkdir(parents=True, exist_ok=True)
    with open(cfg.path.scenes_listeners_file, encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)
    listener_dict = Listener.load_listener_dict(cfg.path.listeners_file)
    enhanced_folder = Path(cfg.path.enhanced_signals)

    # initialize ear
    ear = Ear(**cfg.MSBGEar)

    # Make reference listener with flat audiogram
    listener_ref = Listener(AUDIOGRAM_REF_CLARITY, AUDIOGRAM_REF_CLARITY)

    for scene in tqdm(scenes_listeners):
        for listener_id in scenes_listeners[scene]:
            if enhanced_folder.exists():
                # print('enhanced audio')
                signal_file = enhanced_folder / f"{scene}_HA-output.wav"#zhe li gai cheng wo de
    # for scene in tqdm(scenes_listeners):
    #     for index, listener_id in enumerate(scenes_listeners[scene]):
    #         if enhanced_folder.exists():
    #             # Generate the file name based on the index
    #             signal_file = enhanced_folder / f"{index}_HA-output.wav"
            # if no enhanced signals, use the unprocessed signal for si calculation
            else:
                # signal_file = Path(cfg.path.scenes_folder) / f"{scene}_mixed_CH0.wav"
                # print('mixed audio')
                signal_file = Path(cfg.path.scenes_folder) / f"{scene}_mix_CH1.wav"
            listener = listener_dict[listener_id]

            # signals to write
            outfile_stem = f"{output_path}/{scene}_{listener_id}"
            signal_files_to_write = [
                f"{output_path}/{scene}_flat0dB_HL-output.wav",
                f"{outfile_stem}_HL-output.wav",
                f"{outfile_stem}_HLddf-output.wav",
                f"{outfile_stem}_HL-mixoutput.wav",
            ]
            # if all signals to write exist, pass
            if all(Path(f).exists() for f in signal_files_to_write):
                continue

            signal = read_signal(signal_file)
            mixture_signal = read_signal(
                Path(cfg.path.scenes_folder) / f"{scene}_mix_CH1.wav"
            )

            # Create discrete delta function (DDF) signal for time alignment
            ddf_signal = np.zeros(np.shape(signal))
            ddf_signal[:, 0] = unit_impulse(len(signal), int(MSBG_FS / 2))
            ddf_signal[:, 1] = unit_impulse(len(signal), int(MSBG_FS / 2))

            signals_to_write = [
                listen(ear, ddf_signal, listener_ref),
                listen(ear, signal, listener),
                listen(ear, ddf_signal, listener),
                listen(ear, mixture_signal, listener),
            ]

            for signal, out_signal_file in zip(signals_to_write, signal_files_to_write):
                write_signal(
                    out_signal_file,
                    signal,
                    MSBG_FS,
                    floating_point=True,
                )




@hydra.main(config_path=".", config_name="config")
def run_calculate_SI(cfg: DictConfig) -> None:
    with open(cfg['path']['scenes_listeners_file'], encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)
    proc_folder = Path(cfg['path']['exp_folder']) / "eval_signals"
    sii_file = Path(cfg['path']['exp_folder']) / "sii.csv"
    csv_lines = [["scene", "listener", "sii"]]

    mbstoi_values = []  #

    for scene in tqdm(scenes_listeners):
        for listener in scenes_listeners[scene]:
            logging.info(f"Running SI calculation: scene {scene}, listener {listener}")
            proc = read_signal(proc_folder / f"{scene}_{listener}_HL-output.wav")#  zhege shi du qu de hl xinhao
            clean = read_signal(
                # Path(cfg['path']['scenes_folder']) / f"{scene}_target_anechoic.wav"#zhe li shi target
                Path(cfg['path']['scenes_folder']) / f"{scene}_target_CH1.wav"  # zhe li shi target
            )
            ddf = read_signal(proc_folder / f"{scene}_{listener}_HLddf-output.wav")#hai you ge ddf buzhi dao shisha

            delay = find_delay_impulse(ddf, initial_value=int(MSBG_FS / 2))
            if delay[0] != delay[1]:
                logging.info(f"Difference in delay of {delay[0] - delay[1]}.")
            max_delay = int(np.max(delay))

            if max_delay > 2000:
                logging.error("Error in delay calculation for signal time-alignment.")

            clean_pad = np.zeros((len(clean) + max_delay, 2))
            proc_pad = np.zeros((len(clean) + max_delay, 2))
            # print('proc_pad',len(proc_pad))
            # print('proc',len(proc))

            if len(proc_pad) < len(proc):
                raise ValueError("Padded processed signal is too short.")

            clean_pad[int(delay[0]) : int(len(clean) + int(delay[0])), 0] = clean[:, 0]
            clean_pad[int(delay[1]) : int(len(clean) + int(delay[1])), 1] = clean[:, 1]
            proc_pad[: len(proc)] = proc

            sii = mbstoi(
                clean_pad[:, 0],
                clean_pad[:, 1],
                proc_pad[:, 0],
                proc_pad[:, 1],
                cfg['mbstoi']['sample_rate'],
                cfg['mbstoi']['grid_coarseness'],
            )
            csv_lines.append([scene, listener, sii])
            mbstoi_values.append(sii)

    average_mbstoi = np.mean(mbstoi_values)

    average_mbstoi_file = Path(cfg['path']['exp_folder']) / "average_mbstoi.csv"
    with average_mbstoi_file.open("w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writerow(["average_mbstoi"])
        csv_writer.writerow([average_mbstoi])

    with sii_file.open("w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_lines:
            csv_writer.writerow(line)



def plot_mbstoi_vs_snr(csv_file_path, json_file_path, save_file_path):
    sii_data_speech = []
    snr_data_speech = []
    sii_data_noise = []
    snr_data_noise = []


    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        speech_data = [entry for entry in data if entry.get('interferer', {}).get('type') == 'speech']
        noise_data = [entry for entry in data if entry.get('interferer', {}).get('type') == 'noise']
        for entry in speech_data:
            snr_value = float(entry['SNR'])
            snr_data_speech.extend([snr_value] * 3)

        for entry in noise_data:
            snr_value = float(entry['SNR'])
            snr_data_noise.extend([snr_value] * 3)


    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            if len(row) > 0:
                scene = row[0]
                sii_value = float(row[-1])
                for entry in speech_data:
                    if entry['scene'] == scene:
                        sii_data_speech.append(sii_value)
                        break
                for entry in noise_data:
                    if entry['scene'] == scene:
                        sii_data_noise.append(sii_value)
                        break


    snr_range_start = -5.0
    snr_range_end = 12.5
    snr_range_step = 2.5

    mean_mbstoi_speech = []
    mean_mbstoi_noise = []
    snr_range_values = np.arange(snr_range_start, snr_range_end + snr_range_step, snr_range_step)
    for snr in snr_range_values:
        snr_indices_speech = np.where((np.array(snr_data_speech) >= snr) & (np.array(snr_data_speech) < snr + snr_range_step))[0]
        snr_indices_noise = np.where((np.array(snr_data_noise) >= snr) & (np.array(snr_data_noise) < snr + snr_range_step))[0]
        mbstoi_values_speech = [sii_data_speech[i] for i in snr_indices_speech if i < len(sii_data_speech)]  # Filter sii_data indices
        mbstoi_values_noise = [sii_data_noise[i] for i in snr_indices_noise if i < len(sii_data_noise)]  # Filter sii_data indices
        mean_mbstoi_speech.append(np.mean(mbstoi_values_speech))
        mean_mbstoi_noise.append(np.mean(mbstoi_values_noise))


    print('len snr data (speech):', len(snr_data_speech))
    print('len sii data (speech):', len(sii_data_speech))
    print('len snr data (noise):', len(snr_data_noise))
    print('len sii data (noise):', len(sii_data_noise))

    plt.scatter(snr_data_speech, sii_data_speech, label='MBSTOI (Speech)', color='black',s = 2)
    plt.scatter(snr_data_noise, sii_data_noise, label='MBSTOI (Noise)', color='black',s = 2)
    plt.plot(snr_range_values, mean_mbstoi_speech, color='blue', marker='o', label='Mean MBSTOI SCORE(Speech)')
    plt.plot(snr_range_values, mean_mbstoi_noise, color='red', marker='x', label='Mean MBSTOI SCORE(Noise)')
    plt.xlabel('SNR(dB)')
    plt.ylabel('MBSTOI SCORE')
    plt.title('MBSTOI SCORE vs SNR(dB)')
    plt.legend()
    plt.savefig(save_file_path, dpi=300)
    plt.show()
# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    run_HL_processing()
    run_calculate_SI()
    csv_file_path = "results/sii.csv"
    json_file_path = "scenes.dev.json"
    save_file_path = 'results'
    plot_mbstoi_vs_snr(csv_file_path, json_file_path,save_file_path)
