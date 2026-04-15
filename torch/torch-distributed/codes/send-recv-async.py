import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time  # 添加 time 用于演示异步效果

def init_process(rank, world_size):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        if rank == 0:
            tensor = torch.tensor([100, 200], dtype=torch.float32).cuda()
            print(f"\n=== 初始状态 ===")
            print(f"Rank 0 的初始数据: {tensor}")
            
            # 异步发送数据给 rank 1
            print(f"Rank 0 准备发送数据...")
            send_req = dist.isend(tensor, dst=1)
            print(f"Rank 0 启动异步发送")
            
            # 模拟在等待发送完成时做其他工作
            print(f"Rank 0 正在处理其他任务...")
            time.sleep(1)  # 模拟其他计算任务
            
            # 等待发送完成
            send_req.wait()
            print(f"Rank 0 确认发送完成")
            
        elif rank == 1:
            tensor = torch.zeros(2, dtype=torch.float32).cuda()
            print(f"Rank 1 准备接收数据...")
            
            # 异步接收来自 rank 0 的数据
            recv_req = dist.irecv(tensor, src=0)
            print(f"Rank 1 启动异步接收")
            
            # 模拟在等待接收完成时做其他工作
            print(f"Rank 1 正在处理其他任务...")
            time.sleep(1)  # 模拟其他计算任务
            
            # 等待接收完成
            recv_req.wait()
            print(f"Rank 1 接收完成，数据为: {tensor}")
            
            # 处理数据并异步发送给 rank 2
            tensor = tensor * 2
            print(f"Rank 1 处理后的数据: {tensor}")
            print(f"Rank 1 准备发送数据给 Rank 2...")
            send_req = dist.isend(tensor, dst=2)
            print(f"Rank 1 启动异步发送")
            
            # 模拟在等待发送完成时做其他工作
            print(f"Rank 1 正在处理其他任务...")
            time.sleep(1)  # 模拟其他计算任务
            
            send_req.wait()
            print(f"Rank 1 确认发送完成")
            
        elif rank == 2:
            tensor = torch.zeros(2, dtype=torch.float32).cuda()
            print(f"Rank 2 准备接收数据...")
            
            # 异步接收来自 rank 1 的数据
            recv_req = dist.irecv(tensor, src=1)
            print(f"Rank 2 启动异步接收")
            
            # 模拟在等待接收完成时做其他工作
            print(f"Rank 2 正在处理其他任务...")
            time.sleep(1)  # 模拟其他计算任务
            
            # 等待接收完成
            recv_req.wait()
            print(f"Rank 2 接收完成，最终数据为: {tensor}")
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