import argparse
import os
import re
from glob import glob

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn.functional import conv2d
from torchvision import transforms
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTMAEForPreTraining, ViTMAEModel
from transformers.models.vit_mae.modeling_vit_mae import to_2tuple, get_2d_sincos_pos_embed


def get_embeddings(image, feature_extractor, model):
    '''
    Compute the patchwise embeddings of a PIL image
    using the huggingface feature extractor and model.
    '''

    # convert to RGB
    image = image.convert('RGB')

    # compute outputs of encoder
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)

    # remove CLS token 
    output_seq = outputs.last_hidden_state[:, 1:, :]
    #print(output_seq.shape)

    n_batches, n_patches, n_channels = output_seq.size()
    assert(n_batches == 1)

    # return (n_patches x n_channel) numpy array
    output_seq = output_seq.reshape([n_patches, n_channels])
    return output_seq.detach().numpy()


def get_labels(mask):

    # convert mask to black/white
    mask = transforms.ToTensor()(mask.convert('1'))

    # perform same flattening from 2D to seq, as here
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py#L190

    patch_size = 16 
    kernel = torch.ones([1, 1, patch_size, patch_size])
    
    ratios = conv2d(input=mask, weight=kernel, stride=patch_size) / (patch_size**2)
    label_seq = ratios.flatten(1).reshape((-1))

    return label_seq.numpy()


def load_model(model_str, image_size=400, encoder_only=True):

    if model_str[:7] == 'vit-mae':
        feature_extractor = ViTFeatureExtractor(do_resize=True, image_size=image_size)

        # This does not work: weights are intitiallized randomly     
        # model = ViTMAEModel.from_pretrained(f'facebook/{model_str}', image_size=image_size, mask_ratio=0)

        # Workaround 0: Load AutoEncoder weights and drop decoder
        model = ViTMAEForPreTraining.from_pretrained(f'facebook/{model_str}', mask_ratio=0)
        
        # change image_size to 400 in configuration, sadly doesn't update entire model
        model.config.update({'image_size': image_size})
        config = model.config

        # Workaround 1: update PatchEmbeddings hyperparameters (weights are not changed)
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit_mae/modeling_vit_mae.py#L301
        image_size = to_2tuple(config.image_size)
        patch_size = to_2tuple(config.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

        model.vit.embeddings.patch_embeddings.image_size = image_size
        model.vit.embeddings.patch_embeddings.num_patches = num_patches

        # Workaround 2: update ViTMAEEmbeddings positional embeddings (does not contain trainable weights)
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit_mae/modeling_vit_mae.py#L215
        model.vit.embeddings.num_patches = num_patches
        model.vit.embeddings.position_embeddings = nn.Parameter(
            torch.zeros(1, model.vit.embeddings.num_patches + 1, config.hidden_size), requires_grad=False)
        model.vit.embeddings.config = config
        # model.vit.embeddings.initiallize_weights()
        pos_embed = get_2d_sincos_pos_embed(
            model.vit.embeddings.position_embeddings.shape[-1], 
            int(num_patches**0.5), add_cls_token=True )
        model.vit.embeddings.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Workaround 3: update VitMAEDecoder positional embeddings (does not contain trainable weights)
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit_mae/modeling_vit_mae.py#L741
        # fixed sin-cos embedding
        model.decoder.decoder_pos_embed = nn.Parameter(
                    torch.zeros(1, num_patches + 1, config.decoder_hidden_size), requires_grad=False)  
        model.decoder.config = config
        # model.decoder.initialize_weights(num_patches)
        decoder_pos_embed = get_2d_sincos_pos_embed(
            model.decoder.decoder_pos_embed.shape[-1], 
            int(num_patches**0.5), add_cls_token=True)
        model.decoder.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        if encoder_only:
            model = model.vit

        return feature_extractor, model
    else:
        raise RuntimeError('unkown model string')

def get_dataindices(datapath, dataset, dataprefix):
    image_str = os.path.join(datapath, dataset, 'images', f'{dataprefix}_*.png')
    indices = [int(re.findall(r'\d+', f)[-1]) for f in glob(image_str)]
    return sorted(indices)


def convert_dataset(datapath, dataset, dataprefix, model_str, image_size=400):

    feature_extractor, model = load_model(model_str, image_size=image_size)

    # prepare paths
    indices = get_dataindices(datapath, dataset, dataprefix)
    image_path = os.path.join(datapath, dataset, 'images')
    mask_path = os.path.join(datapath, dataset, 'groundtruth')

    # iterate dataset
    print('Compute embeddings...')
    embs = []
    for idx in tqdm(indices):
        image = Image.open(os.path.join(image_path, f'{dataprefix}_{idx}.png'))
        embs.append(get_embeddings(image, feature_extractor, model))
    
    # gather embedings into dataframe
    embs = np.concatenate(embs)
    df = pd.DataFrame(embs, columns=[f'x{x}' for x in range(embs.shape[1])])
    
    if dataset == 'training':
        print('Compute ratios...')
        lbls = []
        for idx in tqdm(indices):
            mask = Image.open(os.path.join(mask_path, f'{dataprefix}_{idx}.png'))
            mask = mask.resize((image_size, image_size), resample=2)
            lbls.append(get_labels(mask))

        # add labels to dataframe
        lbls = np.concatenate(lbls)
        df['y'] = lbls

    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Convert Dataset to Patchwise ViT-Embeddings and Labels')
    parser.add_argument('--datapath', help='Location of data parent directory')
    parser.add_argument('--dataset', default='training', choices=['training', 'test'], help='String to training/test dataset.')
    parser.add_argument('--dataprefix', default='satimage', help='Prefix of data files')
    parser.add_argument('--model_str', default='vit-mae-base', help='String of pretrained model weights')
    parser.add_argument('--out', hel='Location to store pickle of DataFrame')
    args = parser.parse_args()

    df = convert_dataset(args.datapath, args.dataset, args.dataprefix, args.model_str)
    df.to_pickle(args.out)
    

    