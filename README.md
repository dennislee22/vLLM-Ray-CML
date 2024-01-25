a# vLLM with Ray on CML

<img width="814" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/6a03d5bb-570c-44d8-8bad-55269905962c">

## <a name="toc_0"></a>Table of Contents
[//]: # (TOC)
[1. Objective](#toc_0)<br>
[2. Design Factors](#toc_1)<br>
[3. Deployment Steps](#toc_2)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.1. Create CML Session](#toc_3)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.2. Create Ray (Dashboard+Head) as Application](#toc_4)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.3. Create Flask (Reverse Proxy) as Application](#toc_5)<br>
[4. API Test](#toc_6)<br>
[5. Load Test with Hey](#toc_7)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.1. tensor-parallel-size=1](#toc_8)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.2. tensor-parallel-size=4](#toc_9)<br>
[6. More Experiments](#toc_10)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6.1. tensor-parallel-size=1](#toc_11)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6.2. tensor-parallel-size=2](#toc_12)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6.2. tensor-parallel-size=4](#toc_13)<br>

### <a name="toc_0"></a>1. Objective

- Fitting a huge LLM into a single GPU with limited VRAM during LLM inference is often met with OOM error. Hosting a model with billions of parameters requires thorough understanding of the available ML framework techniques to load the model into the GPU without sacrificing the model precision and output.
- As GPU prices grow exponentially with their size, so chances are companies are more likely to be able to afford multiple smaller GPU devices than a single gigantic one. Data scientists should explore ways to saturate GPU utilization, both VRAM and CUDA/Tensor cores in order to speed up the model inference process.
- While inferencing a model with 7 billion parameters is able to fit into a single GPU device with 40GB of memory, a model with 30 billion parameters needs to leverage on `Tensor Parallelism (TP)` to partition the model weights into the VRAM of all the available GPU devices across multiple nodes. This requires a scalable infrastructure platform.
- This article illustrates simple steps to design a distributed LLM inference solution on a scalable platform with CML (Cloudera Machine Learning) on a Kubernetes platform (Openshift/Rancher).

### <a name="toc_1"></a>2. Design Factors

- Deliver the benefits of Kubernetes to data scientists and yet, shield the complexities of K8s away from them by using SOTA application management or wrapper tool, ie. the ability to spawn multiple worker pods in parallel by utilizing user-friendly dashboard without having to write K8s yaml files. 
- Using a single GPU for a small model inference is likely to achieve low latency but not necessarily high throughput (requests/sec).
- Using multiple nodes with GPU with the help of TP would achieve high throughput but at the expense of low latency. Communications among the TP workers might require high-performance network gadgets to overcome network bottlenecks.
- Select a universally accepted LLM inference and serving engine/framework that supports various types of 🤗 models, e.g. [vLLM](https://docs.vllm.ai/en/latest/models/supported_models.html). It must also support TP, should large model be involved with low specs GPUs. vLLM matches both criteria and it also supports continuous batching ([PagedAttention](https://arxiv.org/abs/2309.06180)) that helps to saturate GPU resources.
- vLLM stores KV cache (gpu-memory-utilization setting) in the GPU memory up to 0.9 (90% of the total capacity). You may allocate lesser percentage with the constrained GPU memory.<br>
<img width="400" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/5e0d84a6-5d51-4052-b3a2-e60b02378296"><br>
- In this architecture, a reverse-proxy service (powered by Flask) is positioned to serve the incoming traffic from external network and traverse the traffic to the vLLM server running as a different pod. vLLM, by default, uses [Ray](https://github.com/ray-project/ray) technology that can scale out the worker pods. Using Ray with CML distributed mechanism is a perfect combo to deliver the scaling capability to AI/ML practitioners. Please check out the simple wrapper scripts in the subsequent topic.
- vLLM can also spin up [OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) to serve model inference using OpenAI API protocol.
- All worker nodes should ideally be using the same NFS storage to share common files, libraries, codes, and model artifacts.

### <a name="toc_2"></a>3. Deployment Steps

- Only 2 Python scripts are required to set up the distributed LLM inference solution.
- [ray_dashboard_4pods.py](ray_dashboard_4pods.py) script is crafted to start the Ray service and appoint the pod as the head. Ray dashboard is also included. vLLM engine will also be initiated in the same pod and it will communicate with Ray head (port 6379) automatically behind the scene. OPENAI compatible server is also started in the same vLLM pod for serving model inference using OpenAI API protocol. Note that 4 worker pods with GPU are configured in this script and you may change the value accordingly.
- [reverse-proxy.py](reverse-proxy.py) script is designed to use Flask as the proxy server. External clients will connect to this frontend and Flask will relay the incoming request to the vLLM pod as the `backend server`.
  
#### <a name="toc_3"></a>3.1 Create CML Session

- Create a new CML project with Python 3.10 and GPU variant.

- Add the following environment variables in the CML project.

<img width="800" alt="image" src="https://github.com/dennislee22/vLLM-Ray-CML/assets/35444414/d630b3d1-41f7-4dd0-b3c4-a01692e4cc45">

- Create a new CML session in the project. The system will communicate with K8s scheduler to spin a pod with the selected resource profile.

<img width="800" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/006ddc97-fbc8-4c92-a9b8-076c51b4c8ee">

- Open the Terminal window in the CML session and install the necessary Python packages.

```
pip install -r requirements.txt
```

- Update the following packages.

```
pip install -U protobuf flask markupsafe jinja2
```

- Clone/download the LFS model in advance. In this case, Vicuna-13b model is selected for testing purpose. Alternatvely, you may also run this as CML job to download the model.

```
git-lfs clone https://huggingface.co/lmsys/vicuna-13b-v1.3
```

- Prepare the scripts in the editor and save them.
<img width="800" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/11578915-b958-4d61-9dd9-24f7f7f3d9af">

#### <a name="toc_4"></a>3.2 Create Ray (Dashboard+Head) as Application

- Start the Ray Dashboard and Head service as the CML application.

<img width="485" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/7fbec7af-c1bd-4af5-bc70-435fb0b12220">

- Upon successful creation, browse the Ray Dashboard URL.

<img width="1432" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/752791a1-8701-4622-af66-a6648d6544e4">

- In the K8s cluster, it shows 4 GPU pods have been provisioned automatically, initiated by the script.

```
$ oc -n cml1-user-5 get pod -o wide
NAME               READY   STATUS    RESTARTS   AGE    IP              NODE        NOMINATED NODE   READINESS GATES
4wr5wz8gfzt0b7s8   5/5     Running   0          54m    10.254.21.215   worker-19   <none>           <none>
8tydoaspor1choqq   5/5     Running   0          54m    10.254.18.73    worker-18   <none>           <none>
u56kjhvvmghgkpi3   5/5     Running   0          168m   10.254.19.56    worker-21   <none>           <none>
uagy60mfci5qqntw   5/5     Running   0          54m    10.254.20.80    worker-20   <none>           <none>
```
  
- Verify the status of the created Ray cluster.

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

#### <a name="toc_5"></a>3.3 Create Flask (Reverse Proxy) as Application

- Start a reverse-proxy service as the CML application for serving the incoming API inference request traffic from the external network.
- In total, 2 CML applications (Flask and Ray Dashboard) should be up and running as shown below.

<img width="800" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/58bcf80d-f549-4ec2-ac0f-364e7f38cf38">

### <a name="toc_6"></a>4. API Test

- Run the following curl command pointing to the reverse-proxy URL to validate the inference result.

```
$ curl https://vllm-api.ml-b5e2c5e4-d7f.apps.field-team-ocp-01.kcloud.cloudera.com/v1/completions -H "Content-Type: application/json" -d '{
"model": "vicuna-13b-v1.3",
"prompt": "Singapore is a",
"max_tokens": 64,
"temperature": 0
}'
{"id":"cmpl-4f49932d923847b695b4ebe5e9494095","object":"text_completion","created":10708810,"model":"vicuna-13b-v1.3","choices":[{"index":0,"text":" small island nation located in Southeast Asia. It is known for its diverse culture, delicious food, and beautiful scenery. The country is a popular tourist destination, attracting millions of visitors each year.\n\nSingapore is a modern city-state with a highly developed economy. It is a","logprobs":null,"finish_reason":"length"}],"usage":{"prompt_tokens":4,"total_tokens":68,"completion_tokens":64}}
```

### <a name="toc_7"></a>5. Load Test with Hey

#### <a name="toc_8"></a>5.1 tensor-parallel-size=1

- Run load test pointing to the reverse-proxy URL with OPENAI compatible API request using [Hey](https://github.com/rakyll/hey)

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

- vLLM log shows successful inference.
```
INFO 01-24 10:56:09 llm_engine.py:706] Avg prompt throughput: 1.6 tokens/s, Avg generation throughput: 43.1 tokens/s, Running: 4 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.2%, CPU KV cache usage: 0.0%
INFO 01-24 10:56:14 llm_engine.py:706] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 45.3 tokens/s, Running: 4 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.3%, CPU KV cache usage: 0.0%
```

#### <a name="toc_9"></a>5.2 tensor-parallel-size=4

- Run load test pointing to the reverse-proxy URL with OPENAI compatible API request using [Hey](https://github.com/rakyll/hey)

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

### <a name="toc_10"></a>6. More Experiments

- Check out the following testing offline inference results that were carried out using [vLLM benchmark throughput](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_throughput.py) and [vLLM benchmark latency](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_latency.py) scripts. The following test cases were done using `open_llama_13b` model.

```
!python benchmark_throughput.py --backend vllm --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json --model open_llama_13b  --num-prompts=100
```
```
!python benchmark_latency.py --tensor-parallel-size 1 --model vicuna-13b-v1.3  --n 10
```

#### <a name="toc_11"></a>6.1 tensor-parallel-size=1

- When loading the `vicuna-13b-v1.3` model into the single GPU node, `nvitop` reports ~60% memory utilization initially.

<img width="700" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/01364d62-d27f-4d94-ae13-e65e375ef1e8">

- And because the `gpu-memory-utilization` is set as 0.7 (default value is 0.9), vLLM will use the balance of 70% of the total memory capacity (after deducting the above ~60% model weights utilization) for KV cache.

<img width="700" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/22203782-3652-4c09-937a-34dffef60bf2">

```
Processed prompts: 100%|██████████████████████| 100/100 [00:43<00:00,  2.32it/s]
Throughput: 2.32 requests/s, 1056.36 tokens/s
```

```
Profiling iterations: 100%|███████████████████████| 3/3 [00:13<00:00,  4.55s/it]
Avg latency: 4.545180882016818 seconds
```

#### <a name="toc_12"></a>6.2 tensor-parallel-size=2

- When loading the `vicuna-13b-v1.3` model into the 2 GPU nodes, `nvitop` reports ~30% memory utilization initially. This is the result of applying `tensor-parallel-size=2` whereby vLLM partitions the model weights and load them into 2 GPU nodes.

<img width="700" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/b83f7aa6-9719-40d2-8602-8ad20f55a274">

- And because the `gpu-memory-utilization` is set as 0.7 (default value is 0.9), vLLM will use the balance of 70% of the total memory capacity (after deducting the above ~30% model weights utilization) for KV cache.

<img width="700" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/f6c1c0b8-1a9a-4558-a646-f0725a22f447">

```
Processed prompts: 100%|██████████████████████| 100/100 [01:41<00:00,  1.02s/it]
Throughput: 0.98 requests/s, 446.75 tokens/s
```

```
Profiling iterations: 100%|███████████████████████| 3/3 [00:42<00:00, 14.09s/it]
Avg latency: 14.08749227412045 seconds
```

#### <a name="toc_13"></a>6.3 tensor-parallel-size=4

- When loading the `vicuna-13b-v1.3` model into the 4 GPU nodes, `nvitop` reports ~16% memory utilization initially. This is the result of applying `tensor-parallel-size=4` whereby vLLM partitions the model weights and load them into 4 GPU nodes.
  
<img width="700" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/ccde5e5a-7830-4cef-a04f-978b6cfbe063">

- And because the `gpu-memory-utilization` is set as 0.7 (default value is 0.9), vLLM will use the balance of 70% of the total memory capacity (after deducting the above ~16% model weights utilization) for KV cache.

<img width="700" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/a70759a3-730b-477f-aa40-6c30b60ca04f">

```
Processed prompts: 100%|██████████████████████| 100/100 [02:49<00:00,  1.70s/it]
Throughput: 0.59 requests/s, 268.08 tokens/s
```

```
Profiling iterations: 100%|███████████████████████| 3/3 [01:11<00:00, 23.71s/it]
Avg latency: 23.70484172180295 seconds
```










