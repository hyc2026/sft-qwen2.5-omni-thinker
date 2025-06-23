# SFT Qwen2.5 Omni Thinker

## Requirements

```bash
pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview
pip install qwen-omni-utils[decord] -U
```

## Dataset

Examples: examples/sft/qwen2.5_omni/train_data.jsonl

(Example data source: [LLaVA-Video178K](https://llava-vl.github.io/blog/2024-09-30-llava-video/))


## Training Scripts

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_omni_trainer \
    data.train_files=examples/sft/qwen2.5_omni/example_data.jsonl \
    data.val_files=examples/sft/qwen2.5_omni/example_data.jsonl \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=Qwen/Qwen2.5-Omni-3B \
    trainer.default_local_dir=./ckpt \
    trainer.total_epochs=1 \
    trainer.project_name=sft-qwen2.5-omni-7b \
    trainer.experiment_name=train \
    model.fsdp_config.model_dtype=bf16
```

## Note

1. Does not support `ulysses_sequence_parallel_size > 1` and `use_remove_padding = True`.
2. It is necessary to ensure that all the data in each **batch** either has videos or not at the same time. The same goes for images and audio.

## Device Recommendation

- 3B: 8 * H100
- 7B: 32 * H100
