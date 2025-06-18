import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def init_process(rank, world_size):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        # 创建初始数据（只在 rank 0 创建有意义的数据）
        if rank == 0:
            tensor = torch.tensor([100, 200], dtype=torch.float32).cuda()
            print(f"\n=== 初始状态 ===")
            print(f"Rank 0 的初始数据: {tensor}")
            # 发送数据给 rank 1
            dist.send(tensor, dst=1)
            print(f"Rank 0 已发送数据到 Rank 1")
            
        elif rank == 1:
            # rank 1 接收来自 rank 0 的数据
            tensor = torch.zeros(2, dtype=torch.float32).cuda()
            dist.recv(tensor, src=0)
            print(f"Rank 1 收到数据: {tensor}")
            
            # 对数据进行修改后发送给 rank 2
            tensor = tensor * 2  # 将数据翻倍
            print(f"Rank 1 处理后的数据: {tensor}")
            dist.send(tensor, dst=2)
            print(f"Rank 1 已发送数据到 Rank 2")
            
        elif rank == 2:
            # rank 2 接收来自 rank 1 的数据
            tensor = torch.zeros(2, dtype=torch.float32).cuda()
            dist.recv(tensor, src=1)
            print(f"Rank 2 收到数据: {tensor}")
            print("\n=== 传输完成 ===")
            
    finally:
        dist.destroy_process_group()

def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    world_size = 3
    print(f"准备启动 {world_size} 个进程...")
    
    mp.spawn(
        init_process,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()