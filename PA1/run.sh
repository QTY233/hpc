#!/bin/bash

# 参数解析
PROGRAM=$1
N=$2
DATAFILE=$3

if [ -z "$PROGRAM" ] || [ -z "$N" ] || [ -z "$DATAFILE" ]; then
    echo "用法: ./run.sh <程序名> <数据规模N> <数据文件路径>"
    echo "示例: ./run.sh odd_even_sort 100 ./data/100.dat"
    exit 1
fi

EXE=./$PROGRAM

# 判断程序是否存在
if [ ! -x "$EXE" ]; then
    echo "错误：程序 $EXE 不存在或不可执行"
    exit 2
fi

# 按规模分档运行
if [ "$N" -lt 110 ]; then
    srun -n 1 "$EXE" "$N" "$DATAFILE"
elif [ "$N" -lt 10010 ]; then
    srun -n 5 "$EXE" "$N" "$DATAFILE"
else
    srun -N 2 -n 56 --cpu-bind=none ./numactl.sh "$EXE" "$N" "$DATAFILE"
fi