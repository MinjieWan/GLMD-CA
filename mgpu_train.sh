OUTPUT_PATH=./output/icoco/train/
LOG_PATH=./log
CONFIG_FILE=./configs/gcoco/defrcn_r101_base.yaml

# create log path
if [ ! -d "$LOG_PATH" ]; then
    mkdir -p "$LOG_PATH"
    echo "Folder $LOG_PATH created."
else
    echo "Folder $LOG_PATH already exists."
fi

nohup python main.py --num-gpus 4 --config-file ${CONFIG_FILE} --dist-url 'tcp://127.0.0.1:50244' --opts OUTPUT_DIR ${OUTPUT_PATH} > ${LOG_PATH}/train.log 2>&1 &