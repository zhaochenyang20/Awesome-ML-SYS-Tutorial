# Dev-log

sppo 相比 Verl 自己实现的 ppo，主要区别是不使用 critic ，以及要修改 loss （传入 rewards）。

plan：

由于ray的monkey patch逻辑可能和单机版不太一样，第一版先侵入式修改了 ppo 的 actor 来验证算法正确性，看 val_score 涨的情况，如果loss实现正确，再考虑 monkey patch 或者自己实现 worker, actor （如果无法实现 monkey patch ）。



当前 step：

1. 创建新的 main_SPPO 与 SPPO Ray Trainer
   
main_sppo
- 基于原有 PPO 的入口做了修改，适配 SPPO 的流程

RaySPPOTrainer
- 最大程度重用 PPO Trainer 逻辑，在 fit 中实现了 SPPO的更新逻辑。

实现了算法正确性验证（val_score 0.78 -> 0.92），接下来打算使代码架构更为合理。

```
main_sppo -> override trainer fit()
          -> fsdp_workers.ActorRolloutRefWorker override init_model() 
	  -> DataParallelPPOActor override update_policy -
 	  -> update_policy
                 
	 
	 -> megatron_workers.ActorRolloutRefWorker needs support ?
```


实现过程中困惑的点：

有两个路径：

1. 根据 Tutorial(https://verl.readthedocs.io/en/latest/advance/dpo_extension.html) 按照 SPPO 数据流的方式自己实现。
2. 学习 PPO 源码，根据 SPPO 与 PPO 的不同（不使用 critic + 修改 loss function），复制 ppo 实现逻辑 + 最小化修改不同点。

最终选择了第 2 步，因为不太了解 PPO  的实现路径上，哪些逻辑对于 verl 是必要的（对 RL 算法 和 Verl 不够熟悉导致的）



