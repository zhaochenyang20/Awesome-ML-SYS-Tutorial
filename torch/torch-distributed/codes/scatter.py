import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_process(rank, world_size):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        # === scatter 示例 ===

        if rank == 0:
            # 在 rank 0 创建要分发的数据
            # 为每个进程准备 2 个数字
            scatter_list = [
                torch.tensor([i * 10, i * 10 + 1], dtype=torch.float32).cuda()
                for i in range(world_size)
            ]
            print("\n=== scatter 前的数据 ===")
            for i, tensor in enumerate(scatter_list):
                print(f"准备发送到 rank {i} 的数据: {tensor.tolist()}")
        else:
            scatter_list = None

        # 准备接收数据的张量
        output_tensor = torch.zeros(2, dtype=torch.float32).cuda()
        print(f"Rank {rank} 初始化接收数据: {output_tensor.tolist()}")
        
        # 执行 scatter 操作
        dist.scatter(output_tensor, scatter_list, src=0)
        
        # 每个进程打印接收到的数据
        print(f"Rank {rank} 收到的数据: {output_tensor.tolist()}")
            
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