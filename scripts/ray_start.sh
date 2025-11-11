export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

HEAD_NODE_IP=172.17.0.12


ray start --head --node-ip-address ${HEAD_NODE_IP} --num-gpus 9 --port 8266 --dashboard-port 8267 --num-cpus 140

