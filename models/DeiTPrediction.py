'''
Description     : DeiT Localized-Answering model
Paper           : CAT-ViL: Co-Attention Gated Vision-Language Embedding for Visual Question Localized-Answering in Robotic Surgery
Author          : Long Bai, Mobarakol Islam, Hongliang Ren
Lab             : Medical Mechatronics Lab, The Chinese University of Hong Kong
Acknowledgement : Code adopted from the official implementation of VisualBERT ResMLP model from 
                  Surgical VQA (https://github.com/lalithjets/Surgical_VQA) and timm/models 
                  (https://github.com/rwightman/pytorch-image-models/tree/master/timm/models).
'''

import torch
from torch import nn
from transformers import VisualBertModel, VisualBertConfig
from timm import create_model
from models.CATViLEmbedding import VisualBertEmbeddings
from utils import *
import torch.nn.functional as F
from models.utils import MLP_adv





class multiresolutionFeatureFusion(nn.Module):
    def __init__( self, n_classes):
        super(multiresolutionFeatureFusion, self).__init__()
        self.dlinear = nn.Linear(384, 768)
        self.postprocess = nn.Sequential(nn.LayerNorm([25,768]), 
                                         nn.Sequential(nn.Linear(768, 512), nn.ReLU(),nn.Linear(512, 384)),
                                         nn.Dropout(p=0.5))

    def forward(self, x_main, x_aux):
        x_aux_L = self.dlinear(x_aux)
        x_main = x_main + x_aux_L
        x_aux = self.postprocess(x_aux_L) + x_aux
        return x_main, x_aux


class DeiTPrediction(nn.Module):
    '''
    Data-Efficient Image Transformer VQLA Model
    vocab_size    = tokenizer length
    encoder_layer = 6
    n_heads       = 8
    num_class     = number of class in dataset
    '''
    def __init__(self, vocab_size, layers, n_heads, num_class, device):
        super(DeiTPrediction, self).__init__()

        self.config = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        self.config.visual_embedding_dim = 512
        self.config.vocab_size = vocab_size 
        self.config.num_hidden_layers = layers
        self.config.num_attention_heads = n_heads      
          
        self.device = device
        self.embeddings = VisualBertEmbeddings(config = self.config)
        self.deit = create_model("deit_base_patch16_224", pretrained=True)     # two ViT, small, base, big

        self.classifier = nn.Linear(768, num_class)
        self.bbox_embed = MLP(768, 768, 4, 3)
        self.mlp = MLP_adv(768, device, 300)
        
    def forward(self, inputs, visual_embeds, if_adv=False):
        # prepare visual embedding
        # append visual features to text
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

        inputs.update({
                        "visual_embeds": visual_embeds,
                        "visual_token_type_ids": visual_token_type_ids,
                        "visual_attention_mask": visual_attention_mask,
                        })
        # Encoder output
        embedding_output = self.embeddings(
            input_ids = inputs['input_ids'].to(self.device),
            token_type_ids = inputs['token_type_ids'].to(self.device),
            position_ids = None,
            inputs_embeds = None,
            visual_embeds = inputs['visual_embeds'].to(self.device),
            visual_token_type_ids = inputs['visual_token_type_ids'].to(self.device),
            image_text_alignment = None,
            if_adv=if_adv
        ) 



        outputs = self.deit.blocks(embedding_output)

        # output
        outputs = self.deit.norm(outputs)
        outputs = outputs.mean(dim=1)             
        outputs_MLP = self.mlp(outputs)
        
        # classification layer        
        classification_outputs = self.classifier(outputs)

        # Detection predition
        bbox_outputs = self.bbox_embed(outputs).sigmoid()
                
        return (outputs_MLP, classification_outputs, bbox_outputs) #wgk