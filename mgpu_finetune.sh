SHOT=10
OUTPUT_PATH=./output/ifsod/finetune_${SHOT}shot
LOG_PATH=./log
CONFIG_FILE=./configs/ifsod/defrcn_ifsod_r101_novel_${SHOT}shot.yaml

# create log path
if [ ! -d "$LOG_PATH" ]; then
    mkdir -p "$LOG_PATH"
    echo "Folder $LOG_PATH created."
else
    echo "Folder $LOG_PATH already exists."
fi

nohup python main.py --num-gpus 4 --config-file $CONFIG_FILE --dist-url 'tcp://127.0.0.1:50111' --opts OUTPUT_DIR $OUTPUT_PATH > ${LOG_PATH}/finetune_${SHOT}shot.log 2>&1 &
