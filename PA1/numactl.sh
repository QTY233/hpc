#!/bin/bash

# LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK # for OpenMPI
# LOCAL_RANK=$MPI_LOCALRANKID # for Intel MPI
LOCAL_RANK=$SLURM_LOCALID # for SLURM

# LOCAL_SIZE=$OMPI_COMM_WORLD_LOCAL_SIZE # for OpenMPI
# LOCAL_SIZE=$MPI_LOCALNRANKS # for Intel MPI
LOCAL_SIZE=$SLURM_TASKS_PER_NODE # for SLURM

NCPUS=$(nproc --all) # eg: 56 or 28 depending on the number of cores
NUM_NUMA=2 # Number of NUMA nodes

# Calculate binding parameters
CORES_PER_PROCESS=$(($NCPUS / $LOCAL_SIZE)) # e.g., 7 when LOCAL_SIZE=4
NUMA_ID=$(($LOCAL_RANK / $NUM_NUMA)) # determine which NUMA node the process belongs to
NUMA_OFFSET=$(($LOCAL_RANK % $NUM_NUMA)) # offset within NUMA node

# Calculate the starting and ending core IDs for each process
CORE_START=$(($NUMA_ID * $CORES_PER_PROCESS * $NUM_NUMA + $NUMA_OFFSET)) # starting core for this rank
CORE_END=$((($NUMA_ID + 1) * $CORES_PER_PROCESS * $NUM_NUMA - $NUM_NUMA + $NUMA_OFFSET)) # ending core for this rank

# Generate a sequence of cores to bind the process to
CORES=$(seq -s, $CORE_START $NUM_NUMA $CORE_END) # e.g., 0,2,4,6,8,10 for NUMA node 0

# Debugging output to check binding info
echo "Process $LOCAL_RANK on $(hostname) bound to core $CORES"

# Execute the program with numactl, binding to the specified cores
exec numactl -C "$CORES" $@