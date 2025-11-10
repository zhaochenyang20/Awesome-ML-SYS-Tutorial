import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_process(rank, world_size):
    try:
        # 1. 加入全局进程组
        torch.cuda.set_device(rank)
        if rank == 0:
            print(f"准备加入全局进程组...")
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        # 2. 创建两个自定义进程组
        group1_ranks = list(range(world_size // 2))
        group2_ranks = list(range(world_size // 2, world_size))
        
        # 初始化累加值为 0
        group1_sum = torch.zeros(1).cuda()
        group2_sum = torch.zeros(1).cuda()
        if rank == 0:
            print(f"组1的初始化累加值: {group1_sum.item()}")
            print(f"组2的初始化累加值: {group2_sum.item()}")
        
        group1 = dist.new_group(group1_ranks)
        group2 = dist.new_group(group2_ranks)
        
        # 3. 在各自的组内进行通信
        tensor = torch.ones(1).cuda() * rank  # 每个进程的输入值为其 rank
        if rank == 0:
            print(f"\n开始进行组内通信...")
        
        if rank == 0:
            print(f"Group1 进行all_reduce操作...")

        # 在对应的组内进行all_reduce，累加结果会更新到 tensor 中
        if rank in group1_ranks:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group1)
            group1_sum = tensor.clone()  # 保存 group1 的累加结果
        else:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group2)
            group2_sum = tensor.clone()  # 保存 group2 的累加结果
        
        # 确保所有进程都能获得两个组的累加结果
        dist.all_reduce(group1_sum, op=dist.ReduceOp.MAX)
        dist.all_reduce(group2_sum, op=dist.ReduceOp.MAX)
        
        if rank == 0:
            print("\n=============== 通信完成 ===============")
            print(f"Group1 (ranks {group1_ranks}): 累加结果为 {group1_sum.item()}")
            print(f"Group2 (ranks {group2_ranks}): 累加结果为 {group2_sum.item()}")
    
    finally:
        dist.destroy_process_group()

def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    world_size = torch.cuda.device_count()
    print(f"准备启动 {world_size} 个进程...")
    
    mp.spawn(
        init_process,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()