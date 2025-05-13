# smoke_test_fixed.py

import os
import time
import torch
import torch.distributed as dist


def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    dist.barrier()
    return local_rank, rank, world_size


def cleanup_ddp():
    dist.barrier()  # final sync before exit
    dist.destroy_process_group()


def main():
    local_rank, rank, world_size = setup_ddp()

    print(
        f"[{time.time():.4f}] Rank {rank}/{world_size} driving GPU {local_rank}",
        flush=True,
    )

    # tiny pause so prints don’t interleave too badly
    time.sleep(0.2)
    dist.barrier()  # sync right before the all-reduce

    # 4) Now the real all-reduce
    print(f"[{rank}] about to all_reduce on device {local_rank}", flush=True)
    tensor = torch.ones(1, device=local_rank) * (rank + 1)
    torch.cuda.synchronize()  # make sure tensor is ready
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    print(f"[{rank}] after all_reduce: {tensor.item()}", flush=True)
    if rank == 0:
        expected = sum(range(1, world_size + 1))
        assert tensor.item() == expected, (
            f"Mismatch: got {tensor.item()}, expected {expected}"
        )
        print("✅ All-reduce sum correct!", flush=True)

    cleanup_ddp()


if __name__ == "__main__":
    main()
