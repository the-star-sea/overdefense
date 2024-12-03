#!/bin/bash

# 如果任意命令出错，脚本将立即退出
set -e

# 设置参数
dataset_list=("oa" "xs")
model_list=("deepseek" "llama3" "mistral" "phi" "gemma" "qwen")
base_dir="/home/zli/SafeDecoding/exaggerated-safety/results"

# 检查 GPU 使用情况的函数
get_free_gpu() {
    for gpu_id in {0..7}; do
        if [[ -z $(nvidia-smi -i $gpu_id --query-compute-apps=pid --format=csv,noheader) ]]; then
            echo $gpu_id
            return
        fi
    done
    echo "No free GPU found"
    exit 1
}

# 根据模型名称获取相应的 model_name 和 template_name
get_model_info() {
    case $1 in
        "deepseek")
            model_name="deepseek-ai/DeepSeek-V2-Lite-Chat"
            template_name="deepseek-chat"
            ;;
        "llama3")
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
            template_name="llama-3"
            ;;
        "mistral")
            model_name="mistralai/Mistral-7B-Instruct-v0.3"
            template_name="mistral"
            ;;
        "phi")
            model_name="microsoft/Phi-3.5-mini-instruct"
            template_name="phi"
            ;;
        "gemma")
            model_name="google/gemma-2-9b-it"
            template_name="gemma"
            ;;
        "qwen")
            model_name="Qwen/Qwen2-7B-Instruct"
            template_name="qwen"
            ;;
        *)
            echo "Invalid model name: $1"
            exit 1
            ;;
    esac
}

# 主循环：遍历数据集和模型
for dataset in "${dataset_list[@]}"; do
    for model in "${model_list[@]}"; do
        get_model_info $model

        # 根据模型名称设置不同的文件检查条件
        if [[ $model == "gemma" ]]; then
            max_files=12
        else
            max_files=13
        fi

        # 检查是否有超过指定数量的匹配文件
        file_count=$(ls $base_dir/${dataset}_${model_name}* 2>/dev/null | wc -l)
        if [[ $file_count -gt $max_files ]]; then
            echo "Skipping ${dataset}_${model_name}, more than $max_files result files found."
            continue
        fi

        # 找到空闲的GPU
        gpu_id=$(get_free_gpu)
        echo "Using GPU: $gpu_id for model: $model_name on dataset: $dataset"

        # 执行 Python 脚本
        CUDA_VISIBLE_DEVICES=$gpu_id python dirty.py --model_name $model --dataset $dataset
    done
done
