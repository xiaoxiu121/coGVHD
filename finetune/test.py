import torch

import torch.nn as nn

embed_dim = 4
num_heads = 2

x = [
  [1, 0, 1, 0], # Input 1
  [0, 2, 0, 2], # Input 2
  [1, 1, 1, 1]  # Input 3
 ]
x = torch.tensor(x, dtype=torch.float32)

w_key = [
  [0, 0, 1, 1],
  [1, 1, 0, 1],
  [0, 1, 0, 1],
  [1, 1, 0, 1]
]
w_query = [
  [1, 0, 1, 1],
  [1, 0, 0, 1],
  [0, 0, 1, 1],
  [0, 1, 1, 1]
]
w_value = [
  [0, 2, 0, 1],
  [0, 3, 0, 1],
  [1, 0, 3, 1],
  [1, 1, 0, 1]
]
w_query = torch.tensor(w_query, dtype=torch.float32)
w_key = torch.tensor(w_key, dtype=torch.float32)
w_value = torch.tensor(w_value, dtype=torch.float32)

querys = x @ w_query 
keys = x @ w_key
values = x @ w_value

multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
attn_output, attn_output_weights = multihead_attn(querys, keys, values)

print(attn_output)

# outputs = block(
#             hidden_states,
#             layer_past=layer_past,
#             rotary_pos_emb=rotary_pos_emb,
#             registered_causal_mask=self.registered_causal_mask,
#             attention_mask=attention_mask,
#             head_mask=head_mask[i],
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#         )
