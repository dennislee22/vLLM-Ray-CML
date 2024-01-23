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

- 1 GPU node
Processed prompts: 100%|██████████████████████| 100/100 [00:43<00:00,  2.32it/s]
Throughput: 2.32 requests/s, 1056.36 tokens/s

- 2 GPU nodes
Processed prompts: 100%|██████████████████████| 100/100 [01:41<00:00,  1.02s/it]
Throughput: 0.98 requests/s, 446.75 tokens/s

- 4 GPU nodes
Processed prompts: 100%|██████████████████████| 100/100 [02:49<00:00,  1.70s/it]
Throughput: 0.59 requests/s, 268.08 tokens/s

<img width="1013" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/256f28e6-d969-4616-ae5d-b46bc93c2faf">

<img width="1432" alt="image" src="https://github.com/dennislee22/vLLM-rayServe/assets/35444414/752791a1-8701-4622-af66-a6648d6544e4">


$ oc -n cml1-user-5 get pod -o wide
NAME               READY   STATUS    RESTARTS   AGE    IP              NODE        NOMINATED NODE   READINESS GATES
4wr5wz8gfzt0b7s8   5/5     Running   0          54m    10.254.21.215   worker-19   <none>           <none>
8tydoaspor1choqq   5/5     Running   0          54m    10.254.18.73    worker-18   <none>           <none>
u56kjhvvmghgkpi3   5/5     Running   0          168m   10.254.19.56    worker-21   <none>           <none>
uagy60mfci5qqntw   5/5     Running   0          54m    10.254.20.80    worker-20   <none>           <none>


