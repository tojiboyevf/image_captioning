import torch
from tqdm.auto import tqdm

def train_model(train_loader, model, loss_fn, acc_fn, optimizer, pad_token=58, desc='', log_interval=25):
    running_acc = 0.0
    running_loss = 0.0
    total_train_words = 0
    model.train()
    t = tqdm(iter(train_loader), desc=f'{desc}')
    for batch_idx, batch in enumerate(t):
        images, captions, lengths = batch
        sort_ind = torch.argsort(lengths, descending=True)
        images = images[sort_ind]
        captions = captions[sort_ind]
        lengths = lengths[sort_ind]

        optimizer.zero_grad()
        output, padding_mask = model(images, captions)
        output = output.permute(1, 2, 0)
        target = torch.cat([captions[:, 1:], pad_token * torch.ones(captions.size(0), 1).type(torch.LongTensor).to(output.device)], dim=1)

        loss = loss_fn(output, target)
        loss_masked = torch.mul(loss, padding_mask)
        final_batch_loss = torch.sum(loss_masked)/torch.sum(padding_mask)
        final_batch_loss.backward()
        optimizer.step()
        running_acc += acc_fn(output, target)
        running_loss += torch.sum(loss_masked).detach().item()
        total_train_words += torch.sum(padding_mask).detach().item()
        t.set_postfix({'loss': running_loss / total_train_words,
                       'acc': running_acc / (batch_idx + 1),
                       }, refresh=True)
        if (batch_idx + 1) % log_interval == 0:
            print(f'{desc} {batch_idx + 1}/{len(train_loader)} '
                  f'train_loss: {running_loss / total_train_words:.4f} '
                  f'train_acc: {running_acc / (batch_idx + 1):.4f}')

    return running_loss / total_train_words
