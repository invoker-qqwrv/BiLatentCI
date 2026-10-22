# BiLatentCI

**Latent-Space Compression for Binaural Speech Enhancement**

![Python](https://img.shields.io/badge/Python-3.9-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange) ![Status](https://img.shields.io/badge/Status-academic%20research-lightgrey)

BiLatentCI is a research implementation investigating **binaural attention speech enhancement**, **latent-space compression**, and **speech-intelligibility-guided numerical optimization**. The goal is to explore lower-bitrate information exchange between two speech-processing branches while preserving binaural speech intelligibility.


> **Research scope:** Cochlear implants motivate this work, but the thesis evaluates the algorithms with the Clarity Enhancement Challenge's hearing-aid and hearing-loss simulation framework. The repository is research code, not a clinically validated implant processor, and energy savings were not measured on hardware.

## Highlights

- Compare **independent**, **front fusion**, **back fusion**, and **binaural fusion** schemes.
- Compress intermediate binaural features with **vector quantization (VQ)**.
- Investigate **EnCodec-inspired convolutional autoencoders** combined with VQ.
- Apply **Kiefer–Wolfowitz (KW)-style numerical optimization** to improve a speech-intelligibility-related objective.
- Evaluate **MBSTOI**, **SI-SDR**, and intelligibility–bitrate trade-offs under a controlled hearing-loss simulation protocol.

## Method overview

The binaural attention model (BinAttse) builds on two Conv-TasNet-style processing branches and uses latent-level attention to exchange complementary acoustic information between the two sides. Compression is introduced at intermediate representations that would otherwise need to be exchanged between processors.

```text
          Noisy left audio                     Noisy right audio
                 |                                     |
             Encoder L                             Encoder R
                 |                                     |
        [Optional AE / VQ]                   [Optional AE / VQ]
                 +---------- Binaural attention ------+
                 |                                     |
         Temporal blocks                       Temporal blocks
                 |                                     |
        [Optional AE / VQ]                   [Optional AE / VQ]
                 +---------- Binaural attention ------+
                 |                                     |
             Decoder L                             Decoder R
                 |                                     |
          Enhanced left                         Enhanced right
                 +---------------+---------------------+
                                 |
                      Hearing-loss simulation
                                 |
                       MBSTOI / SI-SDR analysis
```

This is a **conceptual schematic** summarizing the model families, not a literal implementation diagram for every branch. The thesis explores compression using VQ, EnCodec-inspired autoencoders, and both together. KW numerical optimization is used to improve MBSTOI-related performance, which is not directly handled as a standard differentiable training loss in the study.

## Repository layout

Each folder is a largely standalone experiment with its own `main.py` and supporting utilities.

| Path | Description |
| --- | --- |
| [`diffmode/`](diffmode/) | Fusion comparison: independent, front fusion, back fusion, binaural fusion. |
| [`binattse/`](binattse/) | Binaural attention baseline, evaluation, and KW optimization routines. |
| [`binattse_vq/`](binattse_vq/) | Binaural attention with latent vector quantization. |
| [`binattse_encodec/`](binattse_encodec/) | Binaural attention with an EnCodec-inspired autoencoder. |
| [`binattse_encodecvq/`](binattse_encodecvq/) | Binaural attention with an autoencoder and vector quantization. |
| [`requirements.txt`](requirements.txt) | Pinned dependencies from the original research environment. |
| [`condasetup.slurm`](condasetup.slurm) | Original Slurm/Conda environment setup. |

The repository also contains experiment-specific `config.yaml` or `parameters.yaml` files, MBSTOI evaluation code, sample TFRecords/WAV files, and Slurm job scripts. There is **no root-level training entry point**.

## Thesis results


| System | Nominal bitrate | MBSTOI |
| --- | ---: | ---: |
| OpenMHA baseline | — | 0.31000 |
| Pass-through reference | — | 0.62370 |
| BinAttse (`N=64, S=512`) | Unquantized | 0.58378 |
| BinAttse + VQ | 20 kbit/s | 0.56321 |
| BinAttse + KW | Unquantized | 0.59437 |
| BinAttse + KW + VQ | 20 kbit/s | 0.58970 |
| BinAttse + EnCodec-style AE (`latent dimension=32`) | Unquantized | 0.61327 |
| BinAttse + EnCodec-style AE + VQ | 20 kbit/s | 0.60398 |
| BinAttse + EnCodec-style AE + KW | Unquantized | **0.61754** |
| BinAttse + EnCodec-style AE + KW + VQ | 20 kbit/s | 0.60452 |

The experiments also explore **6, 8, 10, 12, 16, and 20 kbit/s**. The unquantized BinAttse achieves MBSTOI **0.58378**, falling to **0.56321** after VQ at 20 kbit/s. KW optimization improves the unquantized model to **0.59437**. With the 32-dimensional AE, the score increases from **0.61327** to **0.61754** after KW optimization, while AE+VQ retains **0.60398** at 20 kbit/s without KW. In this evaluation, none of these configurations outperforms the pass-through reference.

**Metric caveat:** The thesis finds SI-SDR after hearing-loss simulation strongly affected by the simulated hearing impairment (values around −26 dB for several enhanced conditions). The work therefore emphasizes **MBSTOI for post-simulation intelligibility** and uses SI-SDR primarily on signals before hearing-loss simulation.

## Installation

The historical software stack uses **Python 3.9**, **TensorFlow 2.12.0**, and **PyClarity 0.3.4**. A compatible Linux/GPU environment is recommended.

```bash
git clone https://github.com/invoker-qqwrv/BiLatentCI.git
cd BiLatentCI

conda create -n bilatentci python=3.9 -y
conda activate bilatentci
pip install -r requirements.txt
pip install pyclarity==0.3.4
```

The historical `condasetup.slurm` creates an environment called `clarity5`, downloads Miniconda, and assumes a specific cluster configuration. The original dependencies have **not been revalidated** against newer Python versions or GPUs.

## Dataset

The experiments use the **Clarity Enhancement Challenge (CEC1)** dataset, with binaural target speech, interference signals, room acoustic simulations, and listener-specific hearing-loss profiles. The thesis specifies the following dataset splits:

| Split | Scenes | Speakers |
| --- | ---: | ---: |
| Train | 6,000 | 24 |
| Validation | 2,500 | 10 |
| Test | 1,500 | 6 |

See the [Clarity Challenge](https://claritychallenge.org/) for dataset information and access conditions. The repository contains only some sample audio/TFRecords; **the full thesis dataset is not bundled here**.

The experiment scripts generally expect `data/train.tfr`, `data/valid.tfr`, and `data/test.tfr` relative to the respective experiment directory. Consult `data_generator.py`, the evaluation scripts, and each directory's configuration for the expected schema and supplementary Clarity assets.

## Quick start

Run commands **inside the chosen experiment directory**, because paths are relative. The commands below identify CLI modes present in the source; they have not been independently execution-tested in a fresh environment.

### Train and evaluate BinAttse

```bash
cd binattse
python main.py --mode train --max_epoch 100
python main.py --mode evaluate_test
```

The thesis describes a maximum of **100 epochs** for baseline training, while the current source code defaults to only **3 epochs**. Test evaluation requires a compatible checkpoint.

### Compare fusion strategies

```bash
cd diffmode
python main.py --mode train --model_mode binaural_fusion
```

Available `--model_mode` choices are `independent`, `front_fusion`, `back_fusion`, and `binaural_fusion`. Use the corresponding evaluation mode once the model has been trained.

### Run compression experiments

From the repository root, select one of the relevant experiment directories:

```bash
cd binattse_vq    # or binattse_encodec / binattse_encodecvq
python main.py --mode train --max_epoch 100
python main.py --mode evaluate_test
```

Additional CLI modes include `evaluate_results_models` in the VQ experiments and `predict` in non-VQ experiments. Available artifacts and directory setup vary across experiments.

### Speech intelligibility and KW optimization

Some folders also expose:

```bash
python mbstoi_cal.py
python main.py --mode KW
```

These steps have **implicit prerequisites**: code expects trained weights, pre-generated audio, and paths such as `results/model_best.h5`, `audio/`, and `data/test/`. In the checked-in `binattse/utils.py` and `binattse_encodec/utils.py`, the KW runner's `steps` value is currently **2**, whereas the thesis reports **50 steps**. Inspect the specific `utils.py` before claiming a reproduction.

## Reproducibility and limitations

- **Historical prototype:** Some scripts have cluster-specific absolute paths and Slurm partition names. Edit these for your own environment.
- **Training defaults differ from thesis settings:** review hyperparameters, epochs, random seeds, optimization steps, and checkpoint naming.
- **Dataset coverage:** sample WAV/TFRecord files are not equivalent to all training, validation, and test data described in the thesis.
- **Checkpoints:** evaluation and KW routines may require trained model weights that are not included in the repository.

```

The implementation also builds on research and resources in **Conv-TasNet**, binaural attention speech enhancement, neural audio coding / **EnCodec**, **OpenMHA**, and the **Clarity Enhancement Challenge**. Please acknowledge the underlying publications and data providers when reusing their respective contributions.

## Contact

For project questions, please [open a GitHub issue](https://github.com/invoker-qqwrv/BiLatentCI/issues).
