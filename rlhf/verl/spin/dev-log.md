# Online DPO / SPIN implementation log

## 主要改造内容

### 1. 创建新的 DPO Ray Trainer 与 main_dpo

- **main_spin**  
  创建了一个新的 `main_dpo` 模块，作为在线 DPO 的入口脚本。它负责加载配置、初始化 Ray 集群和构造 DPO 训练器，并启动训练流程。这个模块基于原有 PPO 的入口做了适当修改，以适配在线 DPO 的流程。

- **RayDPOTrainer**  
  新的 `RayDPOTrainer` 重用 PPO 的资源池管理、worker 分组和数据加载逻辑，同时在训练更新阶段调用新的 DPO 更新接口。该接口利用在核心算法(core_algos)中实现的对比损失计算（sigmoid 或 IPO）实现策略模型的直接更新。

### 2. 核心算法模块 (core_algos)  for online dpo

- **损失计算**  
  在 core_algos 中，我们增加了两个新的函数：
  - `compute_online_dpo_loss`（sigmoid 版本）：利用对数概率比的差值，计算出一个基于 sigmoid 的损失。
  - 另一版本为 IPO 损失（基于平方差的版本）。
  
  这两个函数均使用超参数 `beta` 来调控模型更新的幅度，并在返回时输出损失均值。

### 3. 补丁升级 PPO Worker

- **新增 DPO 更新接口**  
  在原有的 DataParallelPPOActor 中，增加了 `update_policy_dpo` 方法。该方法与传统 PPO 更新步骤类似，但它接收通过 union 合并的 chosen 和 rejected 回复，并从 meta_info 中提取 `"chosen_mask"`，随后调用核心算法模块中的 DPO 损失函数计算损失并执行反向传播与梯度更新。
  在原有的 ActorRolloutRefWorker(Worker) 中，增加了 `update_actor_dpo`方法。为`update_policy_dpo`提供接口。

### 4. 一些踩坑记录

- **trainer.ref_update_freq设置**
  之前再完成implement测试的过程中对这个参数的设置一直过大，导致model很容易被reward hack, 进入local max, 从而导致训练崩盘。再重新选择后，已经能够稳定涨点并收敛。
