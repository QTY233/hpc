#!/bin/bash

# LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK # for OpenMPI
# LOCAL_RANK=$MPI_LOCALRANKID # for Intel MPI
LOCAL_RANK=$SLURM_LOCALID # for SLURM

# LOCAL_SIZE=$OMPI_COMM_WORLD_LOCAL_SIZE # for OpenMPI
# LOCAL_SIZE=$MPI_LOCALNRANKS # for Intel MPI
LOCAL_SIZE=$SLURM_TASKS_PER_NODE # for SLURM

NCPUS=$(nproc --all) # eg: 28
NUM_NUMA=2

# calculate binding parameters
# bind to sequential cores in a NUMA domain
CORES_PER_PROCESS=$(($NCPUS / $LOCAL_SIZE)) # eg: 28 when LOCAL_SIZE=28
NUMA_ID=$(($LOCAL_RANK / ($NUM_NUMA * $CORES_PER_PROCESS))) # NUMA node assignment
NUMA_OFFSET=$(($LOCAL_RANK % ($NUM_NUMA * $CORES_PER_PROCESS))) # Offset within NUMA node
CORE_START=$(($NUMA_ID * $CORES_PER_PROCESS * $NUM_NUMA + $NUMA_OFFSET)) 
CORE_END=$((($NUMA_ID + 1) * $CORES_PER_PROCESS * $NUM_NUMA - $NUM_NUMA + $NUMA_OFFSET))
CORES=$(seq -s, $CORE_START $NUM_NUMA $CORE_END)

# execute command with specific cores
echo "Process $LOCAL_RANK on $(hostname) bound to core $CORES"
exec numactl -C "$CORES" $@