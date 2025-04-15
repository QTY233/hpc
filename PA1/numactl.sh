#!/bin/bash

# LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK # for OpenMPI
# LOCAL_RANK=$MPI_LOCALRANKID # for Intel MPI
LOCAL_RANK=$SLURM_LOCALID # for SLURM

# LOCAL_SIZE=$OMPI_COMM_WORLD_LOCAL_SIZE # for OpenMPI
# LOCAL_SIZE=$MPI_LOCALNRANKS # for Intel MPI
TPN=$(echo $SLURM_TASKS_PER_NODE | sed 's/(.*)//')  # 取出括号前的数字
LOCAL_SIZE=$(($SLURM_CPUS_ON_NODE / $TPN))

NCPUS=$(nproc --all) # eg: 28
NUM_NUMA=2

# calculate binding parameters
# bind to sequential cores in a NUMA domain
CORES_PER_PROCESS=$(($NCPUS / $LOCAL_SIZE)) # eg: 7 when LOCAL_SIZE=4
NUMA_ID=$(($LOCAL_RANK * 2)) # eg: 0, 0, 1, 1
CORE_START=$(($NUMA_ID / $LOCAL_SIZE + $NUMA_ID % $LOCAL_SIZE)) # eg: 0, 1, 14, 15
CORES=$(seq -s, $CORE_START $NUM_NUMA $CORE_START) # eg: 0,2,4,6,8,10,12 for rank 0

# execute command with specific cores
echo "Process $LOCAL_RANK on $(hostname) bound to core $CORES, NCMPID=$NCPUS, NUMA_ID=$NUMA_ID, NUMA_OFFSET=$NUMA_OFFSET, LOCAL_SIZE=$LOCAL_SIZE"
# exec numactl -C "$CORES" $@