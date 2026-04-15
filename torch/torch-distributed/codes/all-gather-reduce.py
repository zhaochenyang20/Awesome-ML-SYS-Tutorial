import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def init_process(rank, world_size):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        # 创建测试张量
        tensor = torch.tensor([rank * 10, rank * 10 + 1], dtype=torch.float32).cuda()
        
        # === all_gather 示例 ===
        gathered = [torch.zeros(2, dtype=torch.float32).cuda() for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        
        if rank == 0:
            print("\n=== all_gather 结果 ===")
            print(f"原始张量 (rank 0): {tensor}")
            print("收集所有进程的张量:")
            for i, t in enumerate(gathered):
                print(f"rank {i} 的数据: {t.tolist()}")
        
        # === all_reduce 示例 ===
        reduced_tensor = tensor.clone()  # 创建副本用于 all_reduce
        if rank == 0:
            print(f"before all_reduce: {reduced_tensor}")

        dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
        
        if rank == 0:
            print("\n=== all_reduce 结果 ===")
            print(f"原始张量 (rank 0): {tensor}")
            print(f"归约后的张量 (所有 rank 的和): {reduced_tensor}")
            
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