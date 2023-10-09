#!/bin/bash
#SBATCH --mail-user=wangxiao@tnt.uni-hannover.de
#SBATCH --mail-type=ALL                # Eine Mail wird bei Job-Start/Ende versendet
#SBATCH --job-name=myjob # Name unter dem der Job in der Job-History gespeichert wird
#SBATCH --output=diffmode.txt      # Logdatei r den merged STDOUT/STDERR output (%j wird durch slurm job-ID ersetzt)
#SBATCH --partition=gpu_long_stud#rtition auf der gerechnet werden soll (Bei GPU Jobs unbedingt notwendig)
#SBATCH --time=00:00:00 # hours:minutes:seconds
##SBATCH --array=0-1
#SBATCH --gres=gpu:1                   # Reservierung von einer GPU. Es kann ein bestimmter Typ angefordert werden:
#SBATCH --cpus-per-task=1           # Reservierung von zwei Threads
#SBATCH --mem=10G                      # Reservierung von 1 GB RAM Speicher pro Knoten
#SBATCH --gres=gpu:1


working_dir="/home/wangxiao/tmp/test/diffModel/diffmode"
cd $working_dir

#anacondaroot=/home/wangxiao/anaconda3 #change it to your folder

conda activate /home/wangxiao/tmp/miniconda3/envs/clarity5

srun hostname        
execute_model() {
    local model_mode="$1"


    if [ "$model_mode" == "independent" ]; then


        srun python main.py --mode train --model_mode independent


        srun python main.py --mode evaluate_test

        srun python mbstoi_cal.py

    elif [ "$model_mode" == "back_fusion" ]; then


        srun python main.py --mode train --model_mode back_fusion


        srun python main.py --mode evaluate_test

	srun python mbstoi_cal.py

    elif [ "$model_mode" == "front_fusion" ]; then


        srun python main.py --mode train --model_mode front_fusion


        srun python main.py --mode evaluate_test

	srun python mbstoi_cal.py

    elif [ "$model_mode" == "binaural_fusion" ]; then


        srun python main.py --mode train --model_mode binaural_fusion

        srun python main.py --mode evaluate_test

	srun python mbstoi_cal.py

    else
        echo "invald model_mode: $model_mode"
        exit 1
    fi
}

execute_model "independent"
execute_model "back_fusion"
execute_model "front_fusion"
execute_model "binaural_fusion"