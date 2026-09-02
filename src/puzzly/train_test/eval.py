import torch


def test_step(model, test_dataloader, loss_fn, device, metrics):
    model.eval()
    metrics.reset()

    with torch.inference_mode():
        loss = 0
        for batch, (patches, perm_indices) in enumerate(test_dataloader):
            perm_indices = perm_indices.to(device).long()
            patches = patches.to(device)

            y_logits = model(patches)
            y_preds = y_logits.argmax(dim=1)
            metrics.update(y_preds, perm_indices)

            output = loss_fn(y_logits, perm_indices)
            loss += output.item()

    loss = loss / len(test_dataloader)
    epoch_accuracy = metrics.compute()
    return [epoch_accuracy, loss]