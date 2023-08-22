#!/bin/bash
#SBATCH --mail-user=wangxiao@tnt.uni-hannover.de
#SBATCH --mail-type=ALL                # Eine Mail wird bei Job-Start/Ende versendet
#SBATCH --job-name=norm_each # Name unter dem der Job in der Job-History gespeichert wird
#SBATCH --output=output%.txt      # Logdatei r den merged STDOUT/STDERR output (%j wird durch slurm job-ID ersetzt)
#SBATCH --partition=gpu_long_stud#rtition auf der gerechnet werden soll (Bei GPU Jobs unbedingt notwendig)
#SBATCH --time=100:00:00 # hours:minutes:seconds
##SBATCH --array=0-1
#SBATCH --gres=gpu:1                   # Reservierung von einer GPU. Es kann ein bestimmter Typ angefordert werden:
#SBATCH --cpus-per-task=1           # Reservierung von zwei Threads
#SBATCH --mem=16G                      # Reservierung von 1 GB RAM Speicher pro Knoten
#SBATCH --gres=gpu:1


working_dir="/home/wangxiao/tmp/tiaocan_encodec_quantizer/5"
cd $working_dir
anacondaroot=/home/wangxiao/anaconda3 #change it to your folder

srun hostname        
#source $anacondaroot/envs/activate $binattse

srun python main.py