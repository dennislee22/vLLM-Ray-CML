# vLLM-rayServe

<img width="983" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/7dbe7530-4987-475d-a955-e9f819b6026f">

- Python3.10

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

- 4 GPU nodes (--tensor-parallel-size 4)
```
Profiling iterations: 100%|███████████████████████| 3/3 [01:11<00:00, 23.71s/it]
Avg latency: 23.70484172180295 seconds
```
