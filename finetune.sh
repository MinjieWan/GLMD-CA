SHOT=10
OUTPUT_PATH=./output/ifsod/finetune_${SHOT}shot
LOG_PATH=./log
CONFIG_FILE=./configs/ifsod/defrcn_r101_novel_${SHOT}shot.yaml
GPU_ID=0

# create log path
if [ ! -d "$LOG_PATH" ]; then
    mkdir -p "$LOG_PATH"
    echo "Folder $LOG_PATH created."
else
    echo "Folder $LOG_PATH already exists."
fi

nohup env CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config-file $CONFIG_FILE --opts OUTPUT_DIR $OUTPUT_PATH > ${LOG_PATH}/finetune_${SHOT}shot.log 2>&1 &
