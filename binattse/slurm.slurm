#!/bin/bash
#SBATCH --mail-user=wangxiao@tnt.uni-hannover.de
#SBATCH --mail-type=ALL                # Eine Mail wird bei Job-Start/Ende versendet
#SBATCH --job-name=encodec # Name unter dem der Job in der Job-History gespeichert wird
#SBATCH --output=newversion.txt      # Logdatei r den merged STDOUT/STDERR output (%j wird durch slurm job-ID ersetzt)
#SBATCH --partition=gpu_long_stud#rtition auf der gerechnet werden soll (Bei GPU Jobs unbedingt notwendig)
#SBATCH --time=2:00:00 # hours:minutes:seconds
##SBATCH --array=0-1
#SBATCH --gres=gpu:1                   # Reservierung von einer GPU. Es kann ein bestimmter Typ angefordert werden:
#SBATCH --cpus-per-task=1           # Reservierung von zwei Threads
#SBATCH --mem=8G                      # Reservierung von 1 GB RAM Speicher pro Knoten
#SBATCH --gres=gpu:1


working_dir="/home/wangxiao/tmp/thesis/binattse_tiaohsi"
cd $working_dir
anacondaroot=/home/wangxiao/anaconda3 #change it to your folder

srun hostname        
#source $anacondaroot/envs/activate $binattse

#srun python main.py
execute_mode() {
    if [ "$1" == "train" ]; then
        python main.py --mode train
    elif [ "$1" == "evaluate_test" ]; then
        python main.py --mode evaluate_test
    elif [ "$1" == "KW" ]; then
        python main.py --mode KW
    elif [ "$1" == "predict" ]; then
        python main.py --mode predict
    else
        echo "Invalid mode: $1"
    fi
}

# Execute the desired mode
execute_mode "train"
execute_mode "evaluate_test"
execute_mode "KW"
execute_mode "predict"


srun python mbstoi_cal.py

srun python evaluate.py
