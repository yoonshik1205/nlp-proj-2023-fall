from model import RNN_ENCODER
import torch

model = RNN_ENCODER(5450, 256)

batch_size = 3

captions = torch.randint(1, 5450, (batch_size, 18))
cap_len = torch.randint(1, 18, (batch_size,))
cap_len, _ = torch.sort(cap_len, descending=True)
hidden_state = model.init_hidden(batch_size)

print(cap_len)

out1, out2 = model(captions, cap_len, hidden_state)
print(out1.shape, out2.shape)


# import torch
# import torch.nn as nn

# class CaptionTransformer(nn.Module):
#     def __init__(self, vocab_size, max_len, d_model=256, nhead=8, num_layers=4):
#         super(CaptionTransformer, self).__init__()

#         self.d_model = d_model
#         self.token_embedding = nn.Embedding(vocab_size, d_model)
#         self.positional_encoding = nn.Embedding(max_len, d_model)

#         self.transformer_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
#             num_layers=num_layers
#         )

#     def forward(self, captions, caption_lengths):
#         batch_size, max_len = captions.size()

#         # Token embeddings
#         caption_embedded = self.token_embedding(captions) * (self.d_model ** 0.5)

#         # Positional encoding
#         positions = torch.arange(0, max_len).unsqueeze(0).repeat(batch_size, 1).to(captions.device)
#         position_embedded = self.positional_encoding(positions)

#         # Combine token and positional embeddings
#         encoded_sequence = caption_embedded + position_embedded

#         # Transformer encoder
#         encoded_sequence = encoded_sequence.permute(1, 0, 2)  # Required shape for nn.TransformerEncoder
#         transformer_output = self.transformer_encoder(encoded_sequence)

#         return transformer_output


import torch
import torch.nn as nn
import torch.nn.functional as F

class CaptionTransformer(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=128, nhead=8, num_layers=4):
        super(CaptionTransformer, self).__init__()

        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Embedding(max_len, d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )

    def forward(self, captions, caption_lengths):
        batch_size, max_len = captions.size()
        captions = captions[:, :caption_lengths[0]]
        max_len = caption_lengths[0]
        # Token embeddings
        caption_embedded = self.token_embedding(captions) * (self.d_model ** 0.5)

        # Positional encoding
        positions = torch.arange(0, max_len).unsqueeze(0).repeat(batch_size, 1).to(captions.device)
        position_embedded = self.positional_encoding(positions)

        # Combine token and positional embeddings
        encoded_sequence = caption_embedded + position_embedded

        # Generate mask to exclude padding in captions
        mask = torch.arange(max_len).expand(batch_size, max_len).to(captions.device) >= caption_lengths.unsqueeze(1)

        # Apply padding mask
        encoded_sequence = encoded_sequence.masked_fill(mask.unsqueeze(2), 0)

        # Transformer encoder
        encoded_sequence = encoded_sequence.permute(1, 0, 2)  # Required shape for nn.TransformerEncoder
        transformer_output = self.transformer_encoder(encoded_sequence).permute(1, 2, 0)

        return transformer_output, torch.mean(transformer_output, dim=2)

model2 = CaptionTransformer(5450, 18)

out1, out2 = model2(captions, cap_len)

print(out1.shape, out2.shape)