#!/bin/bash
#SBATCH --mail-user=wangxiao@tnt.uni-hannover.de
#SBATCH --mail-type=ALL                # Eine Mail wird bei Job-Start/Ende versendet
#SBATCH --job-name=condasetup  # Name unter dem der Job in der Job-History gespeichert wird
#SBATCH --output=DAE_out%j.txt      # Logdatei für den merged STDOUT/STDERR output (%j wird durch slurm job-ID ersetzt)
#SBATCH --partition=cpu_short_stud # Partition auf der gerechnet werden soll (Bei GPU Jobs unbedingt notwendig)
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1            # Reservierung von zwei Threads
#SBATCH --mem=5G                       # Reservierung von 1 GB RAM Speicher pro Knoten

working_dir="$(pwd)"
cd $working_dir

srun hostname

mkdir -p ./miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda3/miniconda.sh
bash ./miniconda3/miniconda.sh -b -u -p ./miniconda3
rm -rf ./miniconda3/miniconda.sh

source ./miniconda3/bin/activate

conda create -n BinAttse python=3.6.13

conda activate BinAttse

pip install h5py==3.1.0
pip install keras ==2.4.3
pip install librosa==0.8.0
pip install matplotlib==3.3.3
pip install scipy==1.5.4
pip install tensorflow==2.5.0
pip install tensorflow-gpu==2.4.1
pip install tqdm==4.56.0
pip install xlsxwriter==1.3.8
pip install pyclarity
pip install clarity==0.1.2
pip install numpy==1.19.5