'''
Description     : CAT-ViL Embedding module
Paper           : CAT-ViL: Co-Attention Gated Vision-Language Embedding for Visual Question Localized-Answering in Robotic Surgery
Author          : Long Bai, Mobarakol Islam, Hongliang Ren
Lab             : Medical Mechatronics Lab, The Chinese University of Hong Kong
Acknowledgement : Code adopted from the official implementation of VisualBertModel from 
                  huggingface/transformers (https://github.com/huggingface/transformers.git),
                  GMU (https://github.com/IsaacRodgz/ConcatBERT), and OpenVQA (https://github.com/MILVLG/openvqa).
'''

from torch import nn
from transformers import VisualBertConfig
import torch
import math
import copy
from models.utils import *

def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initializing a VisualBERT visualbert-vqa-coco-pre style configuration
config = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
class VisualBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.hidden_size = 768
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        # For Visual Features
        # Token type and position embedding for image features
        self.visual_token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.visual_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        if config.special_visual_initialize:
            self.visual_token_type_embeddings.weight.data = nn.Parameter(
                self.token_type_embeddings.weight.data.clone(), requires_grad=True
            )
            self.visual_position_embeddings.weight.data = nn.Parameter(
                self.position_embeddings.weight.data.clone(), requires_grad=True
            )

        self.visual_projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)
        self.gated_linear = GatedMultimodalLayer(768*25, 768*25, 768*25)
        self.CMC = CrossModalCalibration(768, nlayers = 1) ##wgk
        mca_hidden_size = 768
        mca_ffn_size = 768
        mca_layers = 6               ##wgk
        self.mca_ed = MCA_ED(mca_hidden_size, mca_ffn_size, mca_layers)

        # self.conv36_visual = torch.nn.Conv1d(in_channels=36, out_channels=25, kernel_size=1)
        # self.conv36_text = torch.nn.Conv1d(in_channels=36, out_channels=25, kernel_size=1)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        visual_embeds=None,
        visual_token_type_ids=None,
        image_text_alignment=None,
        if_adv=False
    ):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        visual_embeds = self.visual_projection(visual_embeds)
        visual_token_type_embeddings = self.visual_token_type_embeddings(visual_token_type_ids)
        visual_position_ids = torch.zeros(
            *visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device
        )
        visual_position_embeddings = self.visual_position_embeddings(visual_position_ids)
        visual_embeddings = visual_embeds + visual_position_embeddings + visual_token_type_embeddings
        # visual embeddings & embeddings (B, 25, 768)
        
        # embeddings = self.conv36_text(embeddings)
        # visual_embeddings = self.conv36_visual(visual_embeddings)
        
        ############################################################################
        visual_embeddings_mask = make_mask(visual_embeddings) ################FINAL       
        embeddings_mask = make_mask(embeddings) ################FINAL
        embeddings, visual_embeddings = self.mca_ed(embeddings, visual_embeddings, embeddings_mask, visual_embeddings_mask) ################FINAL
        visual_embeddings, embeddings = self.mca_ed(visual_embeddings, embeddings, visual_embeddings_mask, embeddings_mask)
        #####Co-attention
        visual_embeddings, embeddings = self.CMC(visual_embeddings, embeddings)
        #####Cross-Modal Calibration
        
        embeddings = torch.flatten(embeddings, start_dim=1, end_dim=-1)
        visual_embeddings = torch.flatten(visual_embeddings, start_dim=1, end_dim=-1)        
        embeddings = self.gated_linear(embeddings, visual_embeddings)  
        embeddings = torch.reshape(embeddings, (-1, 25, 768))
        #####Gated module
        # embeddings = torch.cat((embeddings, visual_embeddings), 1)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MHCrossAttLayer(nn.Module):
    def __init__(self, hidden_dim, nheads, dropout = 0.1):
        super().__init__()
        self.nheads = nheads
        self.hidden_dim = hidden_dim
        hid_hid_dim = hidden_dim//nheads
        self.bottleneck_dim = int(self.hidden_dim)

        self.vision_W3 = nn.Linear(hidden_dim, hid_hid_dim)
        self.vision_sq = nn.Linear(hidden_dim, hid_hid_dim)
        self.vision_ex = nn.Linear(hid_hid_dim, hid_hid_dim)
        self.vision_W3 = _get_clones(self.vision_W3, nheads)
        self.vision_sq = _get_clones(self.vision_sq, nheads)
        self.vision_ex = _get_clones(self.vision_ex, nheads)
        self.vision_W2 = nn.Linear(hidden_dim, self.bottleneck_dim)
        self.vision_W1 = nn.Linear(self.bottleneck_dim, hidden_dim)
        self.vision_LayerNorm = nn.LayerNorm([self.bottleneck_dim,])

        self.semantic_W3 = nn.Linear(hidden_dim, hid_hid_dim)
        self.semantic_sq = nn.Linear(hidden_dim, hid_hid_dim)
        self.semantic_ex = nn.Linear(hid_hid_dim, hid_hid_dim)
        self.semantic_W3 = _get_clones(self.semantic_W3, nheads)
        self.semantic_sq = _get_clones(self.semantic_sq, nheads)
        self.semantic_ex = _get_clones(self.semantic_ex, nheads)
        self.semantic_W2 = nn.Linear(hidden_dim, self.bottleneck_dim)
        self.semantic_W1 = nn.Linear(self.bottleneck_dim, hidden_dim)
        self.semantic_LayerNorm = nn.LayerNorm([self.bottleneck_dim,])

    
    def forward(self, vx, sx):                 #  torch.Size([B, 25, 768])   q->vx; p->sx   (B, 25, 768)

        vx_enhance = []
        for i in range(self.nheads):
            vx_att = torch.sigmoid(self.vision_ex[i](torch.relu(self.vision_sq[i](sx))))
            vx_emb = vx_att * self.vision_W3[i](vx) # Self Aggregation (Initial)
            # vx_emb = vx_att * self.vision_W3[i](sx) # Cross Aggregation
            vx_enhance.append(vx_emb)
        vx_enhance = torch.cat(vx_enhance, dim = -1)   # [torch.Size([B, 25, 384]), torch.Size([B, 25, 384])]  ---->  torch.Size([B, 25, 768])
        vx_enhance = vx + self.vision_W1(torch.relu(self.vision_LayerNorm(self.vision_W2(vx_enhance))))

        sx_enhance = []
        for i in range(self.nheads):
            sx_att = torch.sigmoid(self.semantic_ex[i](torch.relu(self.semantic_sq[i](vx))))
            sx_emb = sx_att * self.semantic_W3[i](sx) # Self Aggregation (Initial)
            # sx_emb = sx_att * self.semantic_W3[i](vx) # Cross Aggregation
            sx_enhance.append(sx_emb)
        sx_enhance = torch.cat(sx_enhance, dim = -1)
        sx_enhance = sx + self.semantic_W1(torch.relu(self.semantic_LayerNorm(self.semantic_W2(sx_enhance))))
        

        return vx_enhance, sx_enhance


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, nheads, dropout = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.nheads = nheads
        self.bilinear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bilinear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bilinear1 = _get_clones(self.bilinear1, nheads)
        self.bilinear2 = _get_clones(self.bilinear2, nheads)
        self.coef = nn.ParameterList([nn.Parameter(torch.ones((hidden_dim, ), dtype = torch.float)/math.sqrt(hidden_dim), requires_grad = True) for _ in range(nheads)])
    
        hid_hid_dim = hidden_dim//nheads
        self.W3 = nn.Linear(hidden_dim, hid_hid_dim)
        self.W3 = _get_clones(self.W3, nheads)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.nonlinear = nn.ReLU(inplace = True)
        self.LayerNorm = nn.LayerNorm([hidden_dim,])
        

    def forward(self, x):
        '''
        x: shape [B, 25, 768]
        '''
        # cal multi-head attention
        x_trans = []
        for i in range(self.nheads):
            x_b1 = self.bilinear1[i](x) # [B, 25, 768]
            x_b2 = self.bilinear2[i](x)
            # x_b1 = torch.sigmoid(self.bilinear1[i](x)) # [B, 25, 768]
            # x_b2 = torch.sigmoid(self.bilinear2[i](x))

            x_b1 = x_b1 * self.coef[i]
            x_att = torch.einsum('abc,abd->abc', x_b1, x_b2)
            x_att = torch.softmax(x_att, dim = -1)
            x_emb = self.W3[i](x)
            x_i = torch.einsum('abc,abf->abf', x_att, x_emb)
            x_trans.append(x_i)  # [B, 25, 768/nheads]
        x_trans = torch.cat(x_trans, dim = -1)
        x_trans = self.W1(self.nonlinear(self.LayerNorm(self.W2(x_trans))))
        x_trans = x + x_trans
        
        return x_trans


class CrossModalCalibration(nn.Module):
    def __init__(self, hidden_dim, nlayers = 1):
        super().__init__()
        self.nlayers = nlayers

        self.CrossAtt = MHCrossAttLayer(hidden_dim, nheads = 2)
        self.CrossAtt = _get_clones(self.CrossAtt, nlayers)

        self.vision_intra_trans = TransformerLayer(hidden_dim, nheads = 2)
        self.semantic_intra_trans = TransformerLayer(hidden_dim, nheads = 2)
        self.vision_intra_trans = _get_clones(self.vision_intra_trans, nlayers)
        self.semantic_intra_trans = _get_clones(self.semantic_intra_trans, nlayers)

    def forward(self, vx, sx):
        '''
        vx: vision features [B, 25, 768]
        sx: semantic features [B, 25, 768]
        '''
        for l in range(self.nlayers):
            # MH2CrossAttLayer_intraTrans2_nlayers1, Highest
            # Inter
            att_vx, att_sx = self.CrossAtt[l](vx, sx)
            # Intra
            vx = self.vision_intra_trans[l](att_vx)
            sx = self.semantic_intra_trans[l](att_sx)
            
        return vx, sx



class GatedMultimodalLayer(nn.Module):
    def __init__(self, size_in1, size_in2, size_out):
        super().__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out

        # Weights hidden state modality 1
        weights_hidden1 = torch.Tensor(size_out, size_in1)
        self.weights_hidden1 = nn.Parameter(weights_hidden1)

        # Weights hidden state modality 2
        weights_hidden2 = torch.Tensor(size_out, size_in2)
        self.weights_hidden2 = nn.Parameter(weights_hidden2)

        # Weight for sigmoid
        weight_sigmoid = torch.Tensor(size_out*2)
        self.weight_sigmoid = nn.Parameter(weight_sigmoid)

        # initialize weights
        nn.init.kaiming_uniform_(self.weights_hidden1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weights_hidden2, a=math.sqrt(5))

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(torch.mm(x1, self.weights_hidden1.t()))
        h2 = self.tanh_f(torch.mm(x2, self.weights_hidden2.t()))
        x = torch.cat((h1, h2), dim=1)
        z = self.sigmoid_f(torch.matmul(x, self.weight_sigmoid.t()))

        return z.view(z.size()[0],1)*h1 + (1-z).view(z.size()[0],1)*h2
