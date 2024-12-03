#!/bin/bash

# 设置文件大小限制（以字节为单位）
SIZE_LIMIT=$((10 * 1024 * 1024))  # 10 MB

# 查找所有文件，排除大于 SIZE_LIMIT 的文件
find . -type f | while read -r file; do
    FILE_SIZE=$(stat -c%s "$file")  # 获取文件大小
    if [ "$FILE_SIZE" -gt "$SIZE_LIMIT" ]; then
        echo "Skipping $file: size is $FILE_SIZE bytes (exceeds limit of $SIZE_LIMIT bytes)"
    else
        git add "$file"
        echo "Added $file"
    fi
done
