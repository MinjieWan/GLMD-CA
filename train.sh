OUTPUT_PATH=./output/gcoco/train/
LOG_PATH=./log
CONFIG_FILE=./configs/gcoco/defrcn_r101_base.yaml
GPU_ID=0

# create log path
if [ ! -d "$LOG_PATH" ]; then
    mkdir -p "$LOG_PATH"
    echo "Folder $LOG_PATH created."
else
    echo "Folder $LOG_PATH already exists."
fi

nohup env CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py --config-file ${CONFIG_FILE} --opts OUTPUT_DIR ${OUTPUT_PATH} > ${LOG_PATH}/train.log 2>&1 &