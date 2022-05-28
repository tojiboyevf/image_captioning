import nltk
import os
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
from torch import stack

def collate_fn_custom(batch):
    images = list()
    captions = list()

    for b in batch:
        images.append(b[0])
        captions.append(b[1])

    images = stack(images, dim=0)

    return images, captions


def CoCoDataloader(transform,
               batch_size=128,
               vocab_threshold=5,
               vocab_file='./vocab_set.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=2,
               img_folder='data/coco/val2014',
               annotations_file='data/coco/annotations/captions_val2014.json'):
    
    

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

    data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=False,
                                      collate_fn = collate_fn_custom,
                                      num_workers=num_workers)

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
        self.ids = list(self.coco.imgToAnns.keys())
        
    def __getitem__(self, index):

        ann_id = self.ids[index]
        captions = []

        for i in range(5):
            captions.append(nltk.word_tokenize(self.coco.imgToAnns[ann_id][i]['caption'].lower()))

        img_id = self.coco.imgToAnns[ann_id][0]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        image = self.transform(image)

        return image, captions

    def __len__(self):
        return len(self.ids)