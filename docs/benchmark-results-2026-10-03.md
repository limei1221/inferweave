## Qwen3-8B (2026-10-03)

| system | arm | rate | done | rejected | preempted | goodput | tok/s | ttft_p99 | tpot_p50 | e2e_p99 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lean-vllm | async-scheduling=False | 1 | 1000 | 0 | 0 | 1.01 | 179 | 0.111 | 0.0072 | 6.716 |
| lean-vllm | async-scheduling=False | 24 | 1000 | 0 | 0 | 19.11 | 3379 | 0.372 | 0.0155 | 13.853 |
| lean-vllm | async-scheduling=False | 32 | 1000 | 0 | 0 | 22.22 | 3930 | 0.922 | 0.0287 | 21.495 |
| lean-vllm | async-scheduling=False | 48 | 1000 | 0 | 0 | 23.35 | 4129 | 7.523 | 0.0450 | 28.698 |
| lean-vllm | async-scheduling=False | 64 | 1000 | 0 | 0 | 23.48 | 4152 | 12.309 | 0.0448 | 29.373 |
| lean-vllm | async-scheduling=True | 1 | 1000 | 0 | 0 | 1.01 | 179 | 0.101 | 0.0069 | 6.418 |
| lean-vllm | async-scheduling=True | 24 | 1000 | 0 | 0 | 19.39 | 3429 | 0.325 | 0.0133 | 11.696 |
| lean-vllm | async-scheduling=True | 32 | 1000 | 0 | 0 | 22.98 | 4064 | 0.862 | 0.0215 | 17.550 |
| lean-vllm | async-scheduling=True | 48 | 1000 | 0 | 0 | 24.81 | 4387 | 5.773 | 0.0408 | 26.552 |
| lean-vllm | async-scheduling=True | 64 | 1000 | 0 | 0 | 25.06 | 4432 | 10.610 | 0.0413 | 27.151 |
| vllm | default | 1 | 1000 | 0 | 0 | 1.01 | 179 | 0.112 | 0.0067 | 6.328 |
| vllm | default | 24 | 1000 | 0 | 0 | 19.40 | 3431 | 0.323 | 0.0127 | 11.435 |
| vllm | default | 32 | 1000 | 0 | 0 | 23.10 | 4085 | 0.816 | 0.0192 | 16.106 |
| vllm | default | 48 | 1000 | 0 | 0 | 25.76 | 4556 | 4.558 | 0.0382 | 24.935 |
| vllm | default | 64 | 1000 | 0 | 0 | 26.09 | 4614 | 9.082 | 0.0389 | 25.539 |


## DeepSeek-V2-Lite-Chat (2026-10-03)

| system | arm | rate | done | rejected | preempted | goodput | tok/s | ttft_p99 | tpot_p50 | e2e_p99 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lean-vllm | default | 1 | 1000 | 0 | 0 | 1.01 | 179 | 0.101 | 0.0065 | 6.516 |
| lean-vllm | default | 24 | 1000 | 0 | 0 | 19.04 | 3366 | 0.268 | 0.0235 | 19.139 |
| lean-vllm | default | 32 | 1000 | 0 | 0 | 22.36 | 3955 | 0.552 | 0.0266 | 20.344 |
| lean-vllm | default | 48 | 1000 | 0 | 0 | 23.18 | 4100 | 6.828 | 0.0446 | 27.344 |
| lean-vllm | default | 64 | 1000 | 0 | 0 | 21.14 | 3739 | 16.476 | 0.0540 | 33.806 |
| vllm | default | 1 | 1000 | 0 | 0 | 1.01 | 179 | 0.088 | 0.0046 | 4.307 |
| vllm | default | 24 | 1000 | 0 | 0 | 19.76 | 3494 | 0.464 | 0.0176 | 15.581 |
| vllm | default | 32 | 1000 | 0 | 0 | 24.57 | 4344 | 3.937 | 0.0199 | 17.264 |
| vllm | default | 48 | 1000 | 0 | 0 | 30.13 | 5328 | 1.515 | 0.0225 | 16.151 |
| vllm | default | 64 | 1000 | 0 | 0 | 31.21 | 5519 | 3.493 | 0.0269 | 18.851 |
