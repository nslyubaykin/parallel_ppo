# Parallel Sampling with [ReLAx](https://github.com/nslyubaykin/relax)
Speeding Up PPO with Parallel Sampling

This repository contains an implementation of PPO algorithm with sampling from parallel environments with ReLAx package.

The performance of single vs multi-thread sampling:

![cpu_sampling](https://github.com/nslyubaykin/parallel_ppo/tree/master/images/cpu_comparison.png)

![cuda_sampling](https://github.com/nslyubaykin/parallel_ppo/tree/master/images/cuda_comparison.png)

Parallel Sampling Takeaways:

1) Avoid small batches. If the tasks are very small, the Ray program can take longer than the equivalent Python program. The issue here is that every task invocation has a non-trivial overhead (e.g., scheduling, inter-process communication, updating the system state) and this overhead dominates the actual time it takes to execute the task.

2) Sampling from GPU hosted policy is slower. If possible, the best option is to run sampling phase on CPU to ensure maximum performance, then on a training phase transfer it back to GPU using actor's .set_device() method. If update phase is computationally cheap, it may be justified to run purely on a CPU.

PPO Humanoid Learning Curve:

![humanoid_learning_curve](https://github.com/nslyubaykin/parallel_ppo/tree/master/images/humanoid_learning_curve.png)

Each x-axis step corresponds to 30k learning transitions batch.
