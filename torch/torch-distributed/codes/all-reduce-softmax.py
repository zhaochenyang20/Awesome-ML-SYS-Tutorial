import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_process(rank, world_size):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        # 创建更复杂的测试张量
        tensor = torch.tensor([rank + 1, rank + 2, rank + 3], dtype=torch.float32).cuda()
        
        if rank == 0:
            print(f"\n初始张量 (rank {rank}): {tensor}")
            
        # 1. 使用 PREMUL_SUM 实现加权和
        weights = torch.tensor([0.3, 0.3, 0.4]).cuda()
        weighted = tensor * weights
        dist.all_reduce(weighted, op=dist.ReduceOp.SUM)
        
        if rank == 0:
            print(f"\n=== 加权和结果 ===")
            print(f"加权后的张量: {weighted}")
            
        # 2. 实现 softmax 的分布式版本
        # 第一步：计算最大值
        max_tensor = tensor.clone()
        if rank == 0:
            print(f"max_tensor before all_reduce: {max_tensor}")
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
        
        if rank == 0:
            print(f"max_tensor after all_reduce: {max_tensor}")
        
        # 第二步：计算 exp(x - max(x))
        exp_tensor = torch.exp(tensor - max_tensor)
        
        # 第三步：计算分母（所有exp的和）
        sum_exp = exp_tensor.clone()
        if rank == 0:
            print(f"sum_exp before all_reduce: {sum_exp}")
        dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM)
        
        if rank == 0:
            print(f"sum_exp after all_reduce: {sum_exp}")
        
        # 第四步：计算最终的 softmax
        softmax_result = exp_tensor / sum_exp
        
        if rank == 0:
            print(f"\n=== 分布式 Softmax 结果 ===")
            print(f"Softmax 结果: {softmax_result}")
            
        # 3. 实现 L2 正则化的分布式版本
        # 第一步：计算平方
        squared = tensor ** 2
        
        # 第二步：求所有元素平方和
        sum_squared = squared.clone()
        if rank == 0:
            print(f"sum_squared before all_reduce: {sum_squared}")
        dist.all_reduce(sum_squared, op=dist.ReduceOp.SUM)
        
        if rank == 0:
            print(f"sum_squared after all_reduce: {sum_squared}")
        
        # 第三步：计算平方根
        l2_norm = torch.sqrt(sum_squared)
        
        # 第四步：正则化
        normalized = tensor / l2_norm
        
        if rank == 0:
            print(f"\n=== 分布式 L2 正则化结果 ===")
            print(f"正则化结果: {normalized}")
            
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