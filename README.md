# deeplabv3plus_DPDDP
This repository is an implementation of deeplabv3+ with dp and ddp.
-- DP : Implements data parallelism at the module level.
This container parallelizes the application of the given module by splitting the input across the specified devices by chunking in the batch dimension (other objects will be copied once per device). In the forward pass, the module is replicated on each device, and each replica handles a portion of the input. During the backwards pass, gradients from each replica are summed into the original module.

-- DDP : DistributedDataParallel (DDP) implements data parallelism at the module level which can run across multiple machines.
