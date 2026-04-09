# Lanty - Lambda Cloud Training

Train Lanty on Lambda Cloud GPU instances for faster iteration on larger datasets.

## Quick Start

1. **Provision an instance** at [cloud.lambda.chat](https://cloud.lambda.chat)
   - Select a GPU instance (see recommendations below)
   - Add your SSH key (~/.ssh/id_ed25519.pub)
   - Note the instance IP when it's ready

2. **Run the deployment script:**
   ```bash
   ./lambda/deploy.sh <INSTANCE_IP>
   ```

3. **When training finishes**, the model is downloaded automatically to `models/`.
   **Terminate your instance** to stop billing!

## Recommended Instances ($100 budget)

| Instance      | GPU       | VRAM  | $/hr  | Hours for $100 | Best for                    |
|---------------|-----------|-------|-------|-----------------|-----------------------------|
| 1x A10        | A10       | 24 GB | $0.60 | ~166 hrs        | Long training runs, budget  |
| 1x A100 SXM   | A100      | 80 GB | $1.10 | ~90 hrs         | Faster training, larger batches |
| 1x H100 SXM   | H100      | 80 GB | $2.49 | ~40 hrs         | Fastest single-GPU          |

**Recommendation:** Start with an **A10** ($0.60/hr) to validate the pipeline, then
switch to **A100** ($1.10/hr) for serious training. The A10 gives you the most
training time for your budget.

## Configuration

Override defaults via environment variables:

```bash
# Use tiny model on A10, 50 epochs
MODEL_SIZE=tiny EPOCHS=50 HOURLY_RATE=0.60 ./lambda/deploy.sh 203.0.113.42

# Use small model on A100, custom seq length
MODEL_SIZE=small EPOCHS=20 SEQ_LEN=256 ./lambda/deploy.sh 203.0.113.42 --rate 1.10
```

| Variable     | Default              | Description                          |
|-------------|----------------------|--------------------------------------|
| LAMBDA_USER | ubuntu               | SSH username                         |
| SSH_KEY     | ~/.ssh/id_ed25519    | SSH private key path                 |
| MODEL_SIZE  | small                | Model config: `tiny` or `small`      |
| EPOCHS      | 20                   | Number of training epochs            |
| SEQ_LEN     | 128                  | Training sequence length             |
| VOCAB_SIZE  | 8192                 | BPE tokenizer vocabulary size        |
| HOURLY_RATE | 1.10                 | Instance cost/hr (for cost estimate) |

## Monitoring Training

The script streams training output in real-time. You can also SSH in directly:

```bash
ssh -i ~/.ssh/id_ed25519 ubuntu@<INSTANCE_IP>
# Check GPU utilization
nvidia-smi
# Watch training output
tail -f /home/ubuntu/lanty/training.log
```

## Resuming Interrupted Training

If training is interrupted, re-run the deploy script with `--resume` support:

```bash
# The script will rsync only changed files, skip tokenizer if already trained,
# and the model trainer supports --resume to continue from the last checkpoint.
./lambda/deploy.sh <INSTANCE_IP>
```

The model saves checkpoints after each epoch (`models/model_epoch_N.bin`),
so you lose at most one epoch of progress.

## Cost Estimates

Rough training time estimates for the full Arch Wiki dataset (~17 MB text):

| Config | A10 (est.) | A100 (est.) | A10 cost | A100 cost |
|--------|-----------|-------------|----------|-----------|
| tiny, 10 epochs  | ~1-2 hrs | ~0.5-1 hr  | ~$1.20  | ~$1.10   |
| small, 10 epochs | ~3-5 hrs | ~1-2 hrs   | ~$3.00  | ~$2.20   |
| small, 50 epochs | ~15-25 hrs | ~5-10 hrs | ~$15.00 | ~$11.00  |

These are rough estimates - actual times depend on tokenizer vocab size,
sequence length, and dataset size.
