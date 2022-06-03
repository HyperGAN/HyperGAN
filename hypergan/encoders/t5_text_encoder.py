
import torch
from torch import nn
import torch.nn.functional as F
from .t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME
class T5TextEncoder(nn.Module):

    def __init__(self, device="cuda:0"):

        super(T5TextEncoder, self).__init__()
        self.t5_dim = get_encoded_dim(DEFAULT_T5_NAME)
        self.max_text_len = 64
        cond_dim=768
        self.text_encoder_name = DEFAULT_T5_NAME
        self.null_text_embed = nn.Parameter(torch.randn(1, self.max_text_len, cond_dim)).cuda()
        self.device = device
    def encode_text(self, text, tokens=64):
        text_embeds, text_masks = t5_encode_text(text, name = self.text_encoder_name)
        text_embeds, text_masks = map(lambda t: t.to(self.device), (text_embeds, text_masks))

        text_tokens = text_embeds
        text_tokens = text_tokens[:, :self.max_text_len]

        text_tokens_len = text_tokens.shape[1]
        remainder = self.max_text_len - text_tokens_len

        if remainder > 0:
            text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))

        #null_text_embed = self.null_text_embed.to(text_tokens.dtype) # for some reason pytorch AMP not working

        #text_tokens = torch.where(
        #    text_keep_mask,
        #    text_tokens,
        #    null_text_embed
        #)
 
        return text_tokens.cuda()


