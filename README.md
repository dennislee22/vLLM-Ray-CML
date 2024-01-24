# vLLM with Ray on CML

<img width="814" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/6a03d5bb-570c-44d8-8bad-55269905962c">

## <a name="toc_0"></a>Table of Contents
[//]: # (TOC)
[1. Objective](#toc_0)<br>
[2. Design Considerations](#toc_1)<br>
[3. Deployment Steps](#toc_2)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.1. Create CML Session](#toc_3)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.2. Create Ray (Dashboard+Head) in CML Application](#toc_4)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.3. Create Flask (Reverse Proxy) in CML Application](#toc_5)<br>
[4. Testing Result](#toc_7)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.1. tensor-parallel-size=1](#toc_8)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.2. tensor-parallel-size=2](#toc_9)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.3. tensor-parallel-size=4](#toc_10)<br>
[5. Load Test with Hey](#toc_11)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.1. tensor-parallel-size=1](#toc_12)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.2. tensor-parallel-size=4](#toc_13)<br>

### <a name="toc_0"></a>1. Objective

- Fitting a huge LLM into a single GPU with limited VRAM during LLM inference is often met with OOM error. Hosting a model with billions of parameters requires thorough understanding of the available ML framework techniques to load the model into the GPU without sacrificing the model precision and output.
- As GPU prices grow exponentially with their size, so chances are companies are more likely to be able to afford multiple smaller GPU devices than a single gigantic one. Data scientists should explore ways to saturate GPU utilization, both VRAM and CUDA/Tensor cores in order to speed up the model inference process.
- While inferencing a model with 7 billion parameters is be able to fit into a single GPU device with 40GB of memory, model with 30 billion parameters needs to involve Tensor Parallelism (TP) to partition the model weights into the VRAM of all the available GPU devices across multiple nodes. This requires a scalable infrastructure platform.
- This article illustrates simple steps to design a distributed LLM inference solution on a scalable platform with CML (Cloudera Machine Learning) on a Kubernetes platform (Openshift/Rancher).

### <a name="toc_1"></a>2. Design Considerations

- Using a single GPU for a small model inference is likely to achieve low latency but not necessarily high throughput (requests/sec).
- Using TP would achieve high throughput but at the expense of low latency. 
- The experiments utilize `batch size=32` configuration for fine-tuning/training the models. Although using higher batch size would increase the training speed, batch size 32 is selected to perform apple-to-apple comparison of the training outcome in terms of training duration and VRAM utilization rate with/without ZeRO technique in place.
- As `t5-large` model has [issue](https://discuss.huggingface.co/t/t5-variants-return-training-loss-0-and-validation-loss-nan-while-fine-tuning/30839) with FP16 during training, FP32 is configured for the experiments. 
- Table below summarizes the benchmark outcome as the result of running the experiments. Each running pod is attached to 1 unit of Nvidia A100-PCIE-40GB device.

| Model     | Technique           | Total Node/Pod | Duration    | epoch  | VRAM (each Pod)   |
| :---      |     :---:           |  :---:         |  ---:       |  :---: |    :---:          |
| t5-small  | w/o deepspeed       |     1          | ~742 secs   |    5   |  3 GB             |
| t5-large  | w/o deepspeed       |     1          | ~7465 secs  |    3   |  15 GB            |
| t5-small  | deepspeed ZeRO-1    |     3          | ~922 secs   |    5   | 5 GB              |
| t5-large  | deepspeed ZeRO-1    |     3          | ~10530 secs |    3   |  13 GB            |
| t5-large  | deepspeed ZeRO-1    |     2          |      -      |    3   | 15 GB             |
| t5-large  | deepspeed ZeRO-3 Offload  |     3    | ~11044 secs |    3   |  9 GB             |
| t5-3b     | w/o deepspeed       |     1          |      -      |    3   |  OOM              |
| t5-3b     | deepspeed ZeRO-3 Offload  |     3    |      N/A    |    3   | 21 GB             |

<sub>OOM = Out-of-Memory</sub>

#### Summary:
- deepspeed `ZeRO-1` with 3 nodes/pods manage to reduce the VRAM consumption when training `t5-large` model, but at the expense of slower training speed compared to single node/pod training without deepspeed.
-  When training LLM in the multi-nodes landscape, the speed is often bottlenecked by network communication overhead (both physical underlay and virtual overlay network) and GPU-CPU-GPU transition process. This can be overcome by resorting to costly options such as SR-IOV and Infiniband technology. Here's the [reference](https://docs.nvidia.com/networking/display/public/sol/rdg+for+accelerating+ai+workloads+in+red+hat+ocp+with+nvidia+dgx+a100+servers+and+nvidia+infiniband+fabric#src-99399137_RDGforAcceleratingAIWorkloadsinRedHatOCPwithNVIDIADGXA100ServersandNVIDIAInfiniBandFabric-OpenShiftContainerPlatformNetworking).
- deepspeed `ZeRO-3 Offload` can exploit both GPU and CPU memory in order to optimize VRAM consumption further compared to `ZeRO-1`. It offloads the optimizer memory and computation from the GPU to the host CPU which is a compelling solution to address memory inefficiency of Adam optimizer. ZeRO Offload uses DeepSpeedCPUAdam which is a highly optimized CPU implementation of Adam, increasing speed by 5-folds compared to standard PyTorch.
- The model size must be significantly huge to take advantage of the deepspeed technology. As seen in `t5-small` model training result, the loaded VRAM is lower than with deepspeed.
- 🤗 trainer code is highly compatible with deepspeed implementation, requires only little code adjustments.

### <a name="toc_2"></a>3. Preparation

- The LLM training in the following experiments use 🤗 Transformers and PyTorch software packages. PyTorch 2.1.2 requires CUDA12.1 as shown below.  
<img width="425" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/d739357e-1421-439d-9395-2bbdf03bbd57"><br>

- The docker image in these experiments, has been installed with [Nvida CUDA nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) version 12.2 for fixing some other incompatibilities.
- As a reference, the outcome of the experiments shows that CUDA nvcc 12.2 can be used as reported in the following training log.
```
Installed CUDA version 12.2 does not match the version torch was compiled with 12.1 but since the APIs are compatible, accepting this combination
```

#### <a name="toc_3"></a>3.1 Build Custom Docker Image

- Build a Docker image locally (based on the native CML image with Jupyter notebook) and push it to the external docker registry, which is represented by Nexus repository in this example.
- The image is installed with the required Nvidia packages. Specific CUDA packages can be referenced from this [Nvidia (ubuntu2004 image)](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/) site.
- For inter-nodes training deployment, deepspeed uses launchers such as OpenMPI and PDSH (a variant of rsh) which are both installed in the docker image as well.

```
docker build -t dlee-deepspeed:2024.1.4 . -f deepspeed-pdsh-mpi-nvcc-jupyter
docker tag dlee-deepspeed:2024.1.4 10.113.204.134:9999/pvcds152/p3.10-nvcc-pdsh-mpi-jptr:2024.1.4
docker push 10.113.204.134:9999/pvcds152/p3.10-nvcc-pdsh-mpi-jptr:2024.1.1
```

- Build another Docker image locally (based on the CML image with Workbench notebook) and push it to the external docker registry. Use this image instead of iPython, if you want to run the training process in the form of CML job.

```
docker build -t dlee-deepspeed:2024.1.4 . -f deepspeed-pdsh-mpi-nvcc-wb
docker tag dlee-deepspeed:2024.1.4 10.113.204.134:9999/pvcds152/p3.10-nvcc-pdsh-mpi-wb:2024.1.4
docker push 10.113.204.134:9999/pvcds152/p3.10-nvcc-pdsh-mpi-wb:2024.1.4
```

- Register the new image in CML.

<img width="800" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/38c82e3c-2ee4-4e00-9fb1-7a2f2c582779"><br>

- Verify that the image has been registered successfully.

<img width="500" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/bdc45baa-54a2-4e39-afa1-7e4ff8988192"><br>

#### <a name="toc_4"></a>3.2 Create CML Session

- Create a new CML project with Python 3.10 and GPU variant.

- Add the newly registered image in the CML project.

<img width="1422" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/a88ca709-a10b-43f1-bd30-b9f6786bafbc"><br>

- Add the following environment variables in the CML project.

<img width="1185" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/299b4736-b9fc-4f91-9f8f-09e52bd25f5d"><br>

- Create a new CML session in the project.
  
<img width="1414" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/0ab49111-1b91-4491-9e81-605822a7f84d"><br>

- Open the Terminal window in the CML session and run the following commands to replace the preconfigured CUDA path with the installed CUDA version in the custom docker image.
  
```
$ rm /usr/local/cuda
$ ln -s /usr/local/cuda-12.2 /usr/local/cuda
$ ls -l /usr/local/cuda
lrwxrwxrwx. 1 cdsw cdsw 20 Jan  4 05:38 /usr/local/cuda -> /usr/local/cuda-12.2
```
- Install the necessary Python packages.

```
pip install -r requirements.txt
```

- Verify the status of deepspeed.

<img width="500" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/abe5a96d-780c-4fe7-b8aa-f943317ec3ff"><br>


#### <a name="toc_5"></a>3.3 Create Tensorboard in CML Application

- Tensorboard is deployed to monitor the training/validation loss. The training logs are serialized and reported to Tensorboard as defined in the training script.
- Create Tensorboard in the CML application
<img width="476" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/f7a42bef-9c1e-4910-a68b-b9b9961ba831">

- Upon successful creation, browse the Tensorboard website.
<img width="571" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/68b4c50e-b536-458e-ad00-7b67716097af">


#### <a name="toc_6"></a>3.4 Prepare Dataset & Model

- In the CML session, run the [prep_dataset.ipynb](prep_dataset.ipynb) to prepare/tokenize the wikiSQL dataset prior to fine-tuning the model.
- You may also opt to clone/download the LFS model in advance.

```
git-lfs clone https://huggingface.co/t5-large
git-lfs clone https://huggingface.co/t5-small
```

### <a name="toc_7"></a>4. Single Node/Pod without ZeRO

<img width="400" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/5e0d84a6-5d51-4052-b3a2-e60b02378296">

- Python3.10

<img width="1411" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/006ddc97-fbc8-4c92-a9b8-076c51b4c8ee">

```
vllm
nvitop
pip install ray[default]
```

```
pip install -U protobuf flask markupsafe jinja2
```

```
git-lfs clone https://huggingface.co/lmsys/vicuna-13b-v1.3
```

<img width="1328" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/11578915-b958-4d61-9dd9-24f7f7f3d9af">

<img width="485" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/7fbec7af-c1bd-4af5-bc70-435fb0b12220">

python -m vllm.entrypoints.openai.api_server --tensor-parallel-size=1 --served-model-name="vicuna-13b-v1.3" --model="vicuna-13b-v1.3" --port=8090 --host="0.0.0.0" > output.log 2>&1 &

```
cdsw@m4u3p7qm2255m2by:~$ ray status --address 10.254.19.61:6379
======== Autoscaler status: 2024-01-23 13:19:25.607925 ========
Node status
---------------------------------------------------------------
Active:
 1 node_10f59d8ac46718137bd6799700ae25f15f7697a7412b5c4cc973fcf4
 1 node_20ef3cf77c59cb49bc2498c5bc3ca7ee953e3cb3add5f8243ddaeb69
 1 node_57c8c66234368fbc1ac5138aeba6e9152cfa6766802b05b8bb3490c0
 1 node_c5994a87133d5b1ee9f574902f4d07f162cac49183243e6cc77ddb51
Pending:
 (no pending nodes)
Recent failures:
 (no failures)

Resources
---------------------------------------------------------------
Usage:
 0.0/160.0 CPU
 0.0/4.0 GPU
 0B/47.58GiB memory
 0B/21.61GiB object_store_memory

Demands:
 (no resource demands)
```


```
$ curl https://vllm-api.ml-b5e2c5e4-d7f.apps.field-team-ocp-01.kcloud.cloudera.com/v1/completions -H "Content-Type: application/json" -d '{
"model": "vicuna-13b-v1.3",
"prompt": "Singapore is a",
"max_tokens": 64,
"temperature": 0
}'
{"id":"cmpl-4f49932d923847b695b4ebe5e9494095","object":"text_completion","created":10708810,"model":"vicuna-13b-v1.3","choices":[{"index":0,"text":" small island nation located in Southeast Asia. It is known for its diverse culture, delicious food, and beautiful scenery. The country is a popular tourist destination, attracting millions of visitors each year.\n\nSingapore is a modern city-state with a highly developed economy. It is a","logprobs":null,"finish_reason":"length"}],"usage":{"prompt_tokens":4,"total_tokens":68,"completion_tokens":64}}
```
- CML applications:

<img width="1414" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/58bcf80d-f549-4ec2-ac0f-364e7f38cf38">



<img width="985" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/01364d62-d27f-4d94-ae13-e65e375ef1e8">

<img width="980" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/22203782-3652-4c09-937a-34dffef60bf2">

<img width="985" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/b83f7aa6-9719-40d2-8602-8ad20f55a274">

<img width="981" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/f6c1c0b8-1a9a-4558-a646-f0725a22f447">

<img width="985" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/ccde5e5a-7830-4cef-a04f-978b6cfbe063">

<img width="981" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/a70759a3-730b-477f-aa40-6c30b60ca04f">


```
!python benchmark_throughput.py --backend vllm --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json --model open_llama_13b  --num-prompts=100
```

- 1 GPU node
```
Processed prompts: 100%|██████████████████████| 100/100 [00:43<00:00,  2.32it/s]
Throughput: 2.32 requests/s, 1056.36 tokens/s
```
- 2 GPU nodes
```
Processed prompts: 100%|██████████████████████| 100/100 [01:41<00:00,  1.02s/it]
Throughput: 0.98 requests/s, 446.75 tokens/s
```
- 4 GPU nodes

```
Processed prompts: 100%|██████████████████████| 100/100 [02:49<00:00,  1.70s/it]
Throughput: 0.59 requests/s, 268.08 tokens/s
```

<img width="1432" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/752791a1-8701-4622-af66-a6648d6544e4">


$ oc -n cml1-user-5 get pod -o wide
NAME               READY   STATUS    RESTARTS   AGE    IP              NODE        NOMINATED NODE   READINESS GATES
4wr5wz8gfzt0b7s8   5/5     Running   0          54m    10.254.21.215   worker-19   <none>           <none>
8tydoaspor1choqq   5/5     Running   0          54m    10.254.18.73    worker-18   <none>           <none>
u56kjhvvmghgkpi3   5/5     Running   0          168m   10.254.19.56    worker-21   <none>           <none>
uagy60mfci5qqntw   5/5     Running   0          54m    10.254.20.80    worker-20   <none>           <none>

```
!python benchmark_latency.py --tensor-parallel-size 1 --model vicuna-13b-v1.3  --n 10
```

- 1 GPU node  (--tensor-parallel-size 1)
```
Profiling iterations: 100%|███████████████████████| 3/3 [00:13<00:00,  4.55s/it]
Avg latency: 4.545180882016818 seconds
```

- 2 GPU nodes (--tensor-parallel-size 2)
```
Profiling iterations: 100%|███████████████████████| 3/3 [00:42<00:00, 14.09s/it]
Avg latency: 14.08749227412045 seconds
```

```
INFO 01-24 10:56:09 llm_engine.py:706] Avg prompt throughput: 1.6 tokens/s, Avg generation throughput: 43.1 tokens/s, Running: 4 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.2%, CPU KV cache usage: 0.0%
INFO 01-24 10:56:14 llm_engine.py:706] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 45.3 tokens/s, Running: 4 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.3%, CPU KV cache usage: 0.0%
```

```
hey -c 5 -m POST -n 50 -H "Content-Type: application/json" -d '{
"model": "vicuna-13b-v1.3",
"prompt": "Singapore is a",
"max_tokens": 64,
"temperature": 0
}' https://vllm-api.ml-b5e2c5e4-d7f.apps.field-team-ocp-01.kcloud.cloudera.com/v1/completions
```

- TP=4
```
$ hey -c 5 -m POST -n 50 -H "Content-Type: application/json" -d '{
"model": "vicuna-13b-v1.3",
"prompt": "Singapore is a",
"max_tokens": 64,
"temperature": 0
}' https://vllm-api.ml-b5e2c5e4-d7f.apps.field-team-ocp-01.kcloud.cloudera.com/v1/completions 

Summary:
  Total:	75.9269 secs
  Slowest:	12.7161 secs
  Fastest:	4.8917 secs
  Average:	7.3568 secs
  Requests/sec:	0.6585
  
  Total data:	27550 bytes
  Size/request:	551 bytes

Response time histogram:
  4.892 [1]	|■
  5.674 [0]	|
  6.457 [29]	|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
  7.239 [4]	|■■■■■■
  8.021 [4]	|■■■■■■
  8.804 [1]	|■
  9.586 [0]	|
  10.369 [7]	|■■■■■■■■■■
  11.151 [2]	|■■■
  11.934 [1]	|■
  12.716 [1]	|■


Latency distribution:
  10% in 6.2370 secs
  25% in 6.2916 secs
  50% in 6.3943 secs
  75% in 8.7868 secs
  90% in 10.2277 secs
  95% in 11.2759 secs
  0% in 0.0000 secs

Details (average, fastest, slowest):
  DNS+dialup:	0.1178 secs, 4.8917 secs, 12.7161 secs
  DNS-lookup:	0.0383 secs, 0.0000 secs, 0.3832 secs
  req write:	0.0001 secs, 0.0000 secs, 0.0003 secs
  resp wait:	7.2386 secs, 4.8914 secs, 11.5366 secs
  resp read:	0.0002 secs, 0.0001 secs, 0.0011 secs

Status code distribution:
  [200]	50 responses
```

- TP=1
```
$ hey -c 5 -m POST -n 50 -H "Content-Type: application/json" -d '{
"model": "vicuna-13b-v1.3",
"prompt": "Singapore is a",
"max_tokens": 64,
"temperature": 0
}' https://vllm-api.ml-b5e2c5e4-d7f.apps.field-team-ocp-01.kcloud.cloudera.com/v1/completions 

Summary:
  Total:	25.1643 secs
  Slowest:	6.2806 secs
  Fastest:	1.9791 secs
  Average:	2.4396 secs
  Requests/sec:	1.9869
  
  Total data:	27550 bytes
  Size/request:	551 bytes

Response time histogram:
  1.979 [1]	|■
  2.409 [41]	|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
  2.839 [2]	|■■
  3.270 [1]	|■
  3.700 [0]	|
  4.130 [0]	|
  4.560 [0]	|
  4.990 [4]	|■■■■
  5.420 [0]	|
  5.850 [0]	|
  6.281 [1]	|■


Latency distribution:
  10% in 2.0408 secs
  25% in 2.0656 secs
  50% in 2.1119 secs
  75% in 2.1767 secs
  90% in 4.7426 secs
  95% in 4.7427 secs
  0% in 0.0000 secs

Details (average, fastest, slowest):
  DNS+dialup:	0.2497 secs, 1.9791 secs, 6.2806 secs
  DNS-lookup:	0.1635 secs, 0.0000 secs, 1.6348 secs
  req write:	0.0001 secs, 0.0000 secs, 0.0002 secs
  resp wait:	2.1895 secs, 1.9789 secs, 3.7841 secs
  resp read:	0.0001 secs, 0.0000 secs, 0.0003 secs

Status code distribution:
  [200]	50 responses
```

- 4 GPU nodes (--tensor-parallel-size 4)
```
Profiling iterations: 100%|███████████████████████| 3/3 [01:11<00:00, 23.71s/it]
Avg latency: 23.70484172180295 seconds
```


