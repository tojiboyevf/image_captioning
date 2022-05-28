import torch
from torch.nn.utils.rnn import pack_padded_sequence
import nltk
from tqdm.auto import tqdm

def train_model(train_loader, model, loss_fn, optimizer,
                acc_fn=lambda source, target: (torch.argmax(source, dim=1) == target).sum().float().item() / target.size(0),
                desc='', log_interval=25):
    running_acc = 0.0
    running_loss = 0.0
    model.train()
    t = tqdm(iter(train_loader), desc=f'{desc}')
    for batch_idx, batch in enumerate(t):
        images, captions, lengths = batch
        sort_ind = torch.argsort(lengths, descending=True)
        images = images[sort_ind]
        captions = captions[sort_ind]
        lengths = lengths[sort_ind]

        optimizer.zero_grad()
        # [sum_len, vocab_size]
        outputs = model(images, captions, lengths)
        # [b, max_len] -> [sum_len]
        targets = pack_padded_sequence(captions, lengths=lengths, batch_first=True, enforce_sorted=True)[0]

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        running_acc += acc_fn(outputs, targets)
        running_loss += loss.item()
        t.set_postfix({'loss': running_loss / (batch_idx + 1),
                       'acc': running_acc / (batch_idx + 1),
                       }, refresh=True)
        if (batch_idx + 1) % log_interval == 0:
            print(f'{desc} {batch_idx + 1}/{len(train_loader)} '
                  f'train_loss: {running_loss / (batch_idx + 1):.4f} '
                  f'train_acc: {running_acc / (batch_idx + 1):.4f}')

    return running_loss / len(train_loader)


def evaluate_model(data_loader, model, bleu_score_fn, tensor_to_word_fn, data='flickr', desc='', device='cpu'):
    running_bleu = [0.0] * 5
    model.eval()
    t = tqdm(iter(data_loader), desc=f'{desc}')
    for batch_idx, batch in enumerate(t):
        if data=='coco':
            images, captions = batch      
        else:
            images, captions, _ = batch
        images = images.to(device)
        outputs = tensor_to_word_fn(model.sample(images).cpu().numpy())

        for i in (1, 2, 3, 4):
            running_bleu[i] += bleu_score_fn(captions, outputs, n=i)
        t.set_postfix({
            'bleu1': running_bleu[1] / (batch_idx + 1),
            'bleu4': running_bleu[4] / (batch_idx + 1),
        }, refresh=True)
    for i in (1, 2, 3, 4):
        running_bleu[i] /= len(data_loader)
    return running_bleu
