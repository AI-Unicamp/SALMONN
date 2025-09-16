#!/bin/bash
#SBATCH --job-name=salmonn_run
#SBATCH --output=/home/pedro.correa/slurm/salmonn_run_%j.out
#SBATCH --error=/home/pedro.correa/slurm/salmonn_run_%j.err
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mail-user=p243236@dac.unicamp.br
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate salmon_env

cd /home/pedro.correa/SALMONN/

python cli_inference_dataset.py --cfg-path configs/decode_config.yaml \
    --input-dir /home/pedro.correa/SALMONN/STYLETTS/7_ref_SERtestSet \
    --output-csv /home/pedro.correa/SALMONN/STYLETTS/styletts_predictions_salmonn_7ref_v4.csv \
    --prompt "Using tone of voice only (prosody: pitch, rhythm, loudness, timbre). Ignore word meaning; do not transcribe. Reply with exactly one: angry | happy | sad | neutral."

