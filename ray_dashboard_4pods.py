import subprocess, sys, os
import cdsw
import ray
import time

DASHBOARD_PORT = os.environ['CDSW_READONLY_PORT']
DASHBOARD_IP = os.environ['CDSW_IP_ADDRESS']
CDSW_APP_PORT=os.environ['CDSW_APP_PORT']

command = "ray start --head --block --include-dashboard=true --dashboard-port=$CDSW_APP_PORT --num-gpus=1 &"
subprocess.run(command, shell = True, executable="/bin/bash")

with open("RAY_HEAD_IP", 'w') as output_file:
    output_file.write(DASHBOARD_IP)
            
ray_head_addr = DASHBOARD_IP + ':6379'

ray_url = f"ray://{DASHBOARD_IP}:10001" 
#ray.init(ray_url)

num_workers=3 #spawn new worker pod apart from this running pod

worker_start_cmd = f"!ray start --block --address={ray_head_addr}"
    
ray_workers = cdsw.launch_workers(
     n=num_workers, 
     cpu=4, 
     memory=16,
     nvidia_gpu=1,
     code=worker_start_cmd,
     )

ray_worker_details = cdsw.await_workers(
     ray_workers, 
     wait_for_completion=False)

os.system("python -m vllm.entrypoints.openai.api_server --tensor-parallel-size=4 --served-model-name=vicuna-13b-v1.3 --model=vicuna-13b-v1.3 --port=5000 --host=0.0.0.0 > vllm.log 2>&1 &")