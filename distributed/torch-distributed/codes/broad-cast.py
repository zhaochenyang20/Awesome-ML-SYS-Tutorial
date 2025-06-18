import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def init_process(rank, world_size):
    try:
        # 初始化进程组
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        # 创建数据
        if rank == 0:
            data1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).cuda()
            data2 = torch.zeros(3).cuda()  # 用于接收rank 1的广播
            print(f"Rank 0 初始数据: data1={data1}, data2={data2}")
        elif rank == 1:
            data1 = torch.zeros(5).cuda()  # 用于接收rank 0的广播
            data2 = torch.tensor([10.0, 20.0, 30.0]).cuda()
            print(f"Rank 1 初始数据: data1={data1}, data2={data2}")
        else:
            data1 = torch.zeros(5).cuda()
            data2 = torch.zeros(3).cuda()
            print(f"Rank {rank} 初始数据: data1={data1}, data2={data2}")
        
        # 先执行rank 0的广播
        dist.broadcast(data1, src=0)
        print(f"Rank {rank} 第一次广播后: data1={data1}")
        print(f"Rank {rank} 第一次广播后: data2={data2}")
        
        # 再执行rank 1的广播
        dist.broadcast(data2, src=1)
        print(f"Rank {rank} 第二次广播后: data1={data1}")
        print(f"Rank {rank} 第二次广播后: data2={data2}")

    finally:
        dist.destroy_process_group()

def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    world_size = torch.cuda.device_count()
    print(f"准备启动 {world_size} 个进程...\n")
    
    mp.spawn(
        init_process,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()