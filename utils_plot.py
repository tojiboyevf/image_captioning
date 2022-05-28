from PIL import Image
from matplotlib import pyplot as plt, cm as cm
from utils_torch import *


def display_images_with_captions(idx2spatial, dset, model, idx2word, metric):
    for idx in idx2spatial:
        x1_y1, x2_y2, x3_y3 = idx2spatial[idx]
        generated_caption = get_picture_caption(idx, dset, model, idx2word)
        score = f"BLEU@4: {metric([[i.split() for i in dset.get_image_captions(idx)[1]]], [generated_caption.split()], n=4):0.2f}"
        img = Image.open(dset.get_image_captions(idx)[0])
        plt.figure(figsize=(12, 6), facecolor="white")
        plt.imshow(img)
        plt.axis('off')
        plt.title(model.name, fontsize=20)
        plt.text(
            *x1_y1, "Gen.: " + generated_caption,
            fontsize=16, bbox=dict(fill=False, edgecolor='black', linewidth=2)
        )
        plt.text(
            *x2_y2, "GT: " + dset.get_image_captions(idx)[1][0],
            fontsize=16, bbox=dict(fill=False, edgecolor='black', linewidth=2)
        )
        plt.text(
            *x3_y3, score,
            fontsize=16,
        )
        plt.show()
