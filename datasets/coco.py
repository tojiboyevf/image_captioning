import nltk
import os
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm


def CoCoDataloader(transform,
               batch_size=1,
               vocab_threshold=None,
               vocab_file='./vocab_set.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               size=0.1,
               num_workers=2,
               img_folder='data/coco/val2014',
               annotations_file='data/coco/annotations/captions_val2014.json',
               shuffle=True,
               random_seed=42):
    
    

    dataset = CoCoDataset(transform=transform,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)

    indices = dataset.get_train_indices()

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(size * dataset_size))

    if shuffle :
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    indices = indices[:split]

    initial_sampler = data.sampler.SubsetRandomSampler(indices)

    data_loader = data.DataLoader(dataset=dataset, 
                                    num_workers=num_workers,
                                    batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                            batch_size=dataset.batch_size,
                                                                            drop_last=False))

    return data_loader

class CoCoDataset(data.Dataset):
    
    def __init__(self, transform, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, annotations_file, vocab_from_file, img_folder):
        self.transform = transform
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder

        self.coco = COCO(annotations_file)
        self.ids = list(self.coco.anns.keys())
        print('Obtaining caption lengths...')
        all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))]
        self.caption_lengths = [len(token) for token in all_tokens]

        
    def __getitem__(self, index):

        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        # orig_image = np.array(image)
        image = self.transform(image)

        return image, caption

    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        return len(self.ids)
