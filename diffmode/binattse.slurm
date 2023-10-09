#!/bin/bash
#SBATCH --mail-user=wangxiao@tnt.uni-hannover.de
#SBATCH --mail-type=ALL                # Eine Mail wird bei Job-Start/Ende versendet
#SBATCH --job-name=myjob # Name unter dem der Job in der Job-History gespeichert wird
#SBATCH --output=output.txt      # Logdatei r den merged STDOUT/STDERR output (%j wird durch slurm job-ID ersetzt)
#SBATCH --partition=gpu_long_stud#rtition auf der gerechnet werden soll (Bei GPU Jobs unbedingt notwendig)
#SBATCH --time=00:00:00 # hours:minutes:seconds
##SBATCH --array=0-1
#SBATCH --gres=gpu:1                   # Reservierung von einer GPU. Es kann ein bestimmter Typ angefordert werden:
#SBATCH --cpus-per-task=1           # Reservierung von zwei Threads
#SBATCH --mem=10G                      # Reservierung von 1 GB RAM Speicher pro Knoten
#SBATCH --gres=gpu:1


working_dir="/home/wangxiao/tmp/binattse_diff_mode"
cd $working_dir
anacondaroot=/home/wangxiao/anaconda3 #change it to your folder

srun hostname        
execute_model() {
    local model_mode="$1"
    

    if [ "$model_mode" == "front_fusion" ]; then
        
        
        srun python main.py --mode train --model_mode front_fusion
        
        
        srun python main.py --mode evaluate_test
	srun python mbstoi_cal.py

    elif [ "$model_mode" == "independent" ]; then
        
        
        srun python main.py --mode train --model_mode independent
        
        
        srun python main.py --mode evaluate_test
    else
        echo "Invalid model mode: $model_mode"
        exit 1
    fi
}

# 使用示例

execute_model "independent" 