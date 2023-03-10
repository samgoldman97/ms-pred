#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=hyperopt_ms
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/hyperopt_%j.log
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --nodelist=node[1236,1237]
#SBATCH --time=0-00:30:00
#SBATCH --partition=sched_mit_ccoley

##SBATCH --tasks-per-node=1
##SBATCH --mem-per-cpu=2GB


source /etc/profile 
source /home/samlg/.bashrc

ray_var=~/miniconda3/envs/ms-gen/bin/ray
hostname=/bin/hostname
conda activate ms-gen
unset LD_PRELOAD

set -x

# __doc_head_address_start__


# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
list_len=${#nodes_array[@]}
if [[ $list_len -gt 0 ]]; then
    head_node=${nodes_array[0]}
else 
    head_node=$nodes
fi

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" $hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi
# __doc_head_address_end__

# __doc_head_ray_start__
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
echo "Slurm gpus: $SLURM_JOB_GPUS"
ngpus=`echo $SLURM_JOB_GPUS | grep -P -o '\d' | wc -l`

srun --nodes=1 --ntasks=1 -w "$head_node" \
    $ray_var start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${ngpus}" --block &
#srun --nodes=1 --ntasks=1 -w "$head_node" \
#    $ray_var start --head --node-ip-address="$head_node_ip" --port=$port \
#    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${ngpus}" --block &
# __doc_head_ray_end__

# __doc_worker_ray_start__
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES))

for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        $ray_var start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${ngpus}" --block &
    sleep 5
done
# __doc_worker_ray_end__


# __doc_script_start__
# ray/doc/source/cluster/examples/simple-trainer.py
eval $CMD
