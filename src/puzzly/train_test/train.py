import time


def train_step(model, train_dataloader, loss_fn, optimizer, device, metrics, gpu_transform, log_every=100):
    print("NUMBER OF BATCHES:", len(train_dataloader))
    model.train()
    metrics.reset()
    loss = 0
    total_batches = len(train_dataloader)
    epoch_start = time.time()

    for batch, (patches, perm_indices) in enumerate(train_dataloader):
        if batch % log_every == 0:
            elapsed = time.time() - epoch_start
            pct = batch / total_batches * 100
            rate = batch / elapsed if elapsed > 0 else 0
            print(f"Training: {batch}/{total_batches} ({pct:.1f}%) | {elapsed:.1f}s elapsed | {rate:.2f} batches/sec")

        perm_indices = perm_indices.to(device).long()
        patches = patches.to(device)
        batch_size = patches.size(0)
        patches = patches.view(-1, *patches.shape[2:])
        patches = gpu_transform(patches)
        patches = patches.view(batch_size, 9, *patches.shape[1:])

        y_logits = model(patches)
        y_preds = y_logits.argmax(dim=1)
        metrics.update(y_preds, perm_indices)

        output = loss_fn(y_logits, perm_indices)
        loss += output.item()

        optimizer.zero_grad()
        output.backward()
        optimizer.step()

    epoch_accuracy = metrics.compute()
    loss = loss / len(train_dataloader)
    return [epoch_accuracy, loss]