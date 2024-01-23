# vLLM-rayServe


- Python3.10

```
vllm
nvitop
pip install ray[default]
```

```
pip install -U protobuf
```

```
git-lfs clone https://huggingface.co/lmsys/vicuna-13b-v1.3
```


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
<img width="1013" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/256f28e6-d969-4616-ae5d-b46bc93c2faf">

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
