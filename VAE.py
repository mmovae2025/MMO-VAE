import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import re 


class Encoder(nn.Module):
    def __init__(self,
                 embed_dim,
                 hidden_dim,
                 latent_dim,
                 num_layers):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=embed_dim, # 256
                          hidden_size=hidden_dim, # 512
                          num_layers=num_layers, # 1
                          batch_first=True, # 2
                          bidirectional=True,
                          dropout=0
                          )

        self.mu = nn.Linear(2 * hidden_dim, latent_dim) # 2 * 512 -> 48
        self.std = nn.Sequential(
            nn.Linear(2 * hidden_dim, latent_dim), # 2 * 512 -> 48
            nn.Softplus())

    def forward(self, embed_x, lengths, label = None, prop=None):
        packed = pack_padded_sequence(embed_x, lengths, batch_first=True)
        output, hidden = self.gru(packed)
    
        forward_x_z = hidden[0::2]  # forward
        backward_x_z = hidden[1::2]  # backward

        forward_mean = forward_x_z.mean(dim=0)
        backward_mean = backward_x_z.mean(dim=0)

        x_z = torch.cat([forward_mean, backward_mean], dim=1)
    
        mu = self.mu(x_z)
        std = self.std(x_z)

        return mu, std

        

class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim, value_range= None):
        super(Predictor, self).__init__()
        self.latent_layer_1 = nn.Linear(input_dim, 2*input_dim)
        self.latent_layer_2 = nn.Linear(2*input_dim, 2*input_dim)
        self.latent_layer_3 = nn.Linear(2*input_dim, 2*input_dim)
        self.latent_layer_4 = nn.Linear(2*input_dim, output_dim)

        self.value_range = value_range
        if value_range is not None:
            self.min_val, self.max_val, self.closed_left, self.closed_right = self.parse_value_range(value_range)

    

    def parse_value_range(self, value_range):
        match = re.match(r'(\(|\[)(.*),(.*)(\)|\])', value_range)
        if not match:
            raise ValueError("Invalid range format. Use (min,max), (min,max], [min,max), or [min,max]")
        
        closed_left = match.group(1) == '['
        closed_right = match.group(4) == ']'
        min_val = float(match.group(2).strip())
        max_val = float(match.group(3).strip())
        
        return min_val, max_val, closed_left, closed_right

    
    def forward(self, z): 
        h = F.selu(self.latent_layer_1(z))
        h = F.selu(self.latent_layer_2(h))
        h = F.selu(self.latent_layer_3(h))
        y = self.latent_layer_4(h)
        
        # 범위에 따라 출력값을 조절
        if self.value_range is not None:
            y = torch.sigmoid(y)
            if not self.closed_left:
                min_val = self.min_val + 1e-6  
            else:
                min_val = self.min_val
                
            if not self.closed_right:
                max_val = self.max_val - 1e-6  
            else:
                max_val = self.max_val
                
            y = y * (max_val - min_val) + min_val
        return y




class Decoder(nn.Module):
    def __init__(self,
                 embed_dim,
                 hidden_dim,
                 latent_dim,
                 num_layers,
                 dropout = 0.1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.bridge = nn.Linear(latent_dim, (num_layers * hidden_dim), bias=True)
        self.decoder_gru = nn.GRU(input_size=embed_dim,
                                  hidden_size=hidden_dim,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  dropout=0)

        self.pre_output = nn.Linear(embed_dim + hidden_dim, hidden_dim, bias=True)

    def forward(self, trg_embed, z, max_len, generator, embeder, sos_token=None, tf_ratio = 1.0, decode_sampling = False):
        batch_size = z.size(0)
        hidden = self.bridge(z) # z : [batch_size, latent_dim] -> [batch_size, num_layers * hidden_dim]
        hidden = hidden.reshape(batch_size, self.num_layers, self.hidden_dim) #
        hidden = hidden.permute(1, 0, 2) # [num_layers, batch_size, latent_dim]
        hidden = torch.tanh(hidden).contiguous() # [num_layers, batch_size, latent_dim]
        pre_output_vectors = []

        #Setting Start Token 
        if sos_token is not None:
            prev_embed = sos_token
        else: 
            prev_embed = trg_embed[:, 0].unsqueeze(1) # trg_embed : [batch_size, max_len, embed_dim] -> [batch_size, 1, embed_dim]

        for i in range(max_len): 
            pre_output, hidden = self.forward_step(prev_embed, hidden)
            pre_output_vectors.append(pre_output)
            if i is max_len - 1:
                break

            if tf_ratio > torch.rand(1).item():
                if decode_sampling:
                    token_logits = generator(pre_output.squeeze(1))
                    token_probs = torch.softmax(token_logits, dim=-1)
                    token = torch.multinomial(token_probs, 1).squeeze(1) 
                    embed_token = embeder(token)
                    prev_embed = embed_token.unsqueeze(1)
                else:
                    token = generator(pre_output.squeeze(1))
                    token = torch.max(token, dim=-1)[1]
                    embed_token = embeder(token)
                    prev_embed = embed_token.unsqueeze(1)

            
            else:
                prev_embed = trg_embed[:, i].unsqueeze(1) # trg_embed : [batch_size, max_len, embed_dim] -> [batch_size, 1, embed_dim]
        
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return pre_output_vectors

            
    def forward_step(self, prev_embed, hidden):
        #output -> [batch, 1, hidden_dim]
        #hidden -> [num_layer, batch, hidden_dim]
        output, hidden = self.decoder_gru(prev_embed, hidden)

        # input-feeding approach
        pre_output = torch.cat([prev_embed, output], dim = 2) # [batch, 1, embed_dim + hidden_dim]
        pre_output = self.pre_output(pre_output)  # embed_din + hidden_dim -> hidden_dim
        return pre_output, hidden

    def generate(self, z, max_len, generator, embeder, sos_token):
        batch_size = z.size(0)
        hidden = self.bridge(z) # z : [batch_size, latent_dim] -> [batch_size, num_layers * hidden_dim]
        hidden = hidden.reshape(batch_size, self.num_layers, self.hidden_dim) #
        hidden = hidden.permute(1, 0, 2) # [num_layers, batch_size, latent_dim]
        hidden = torch.tanh(hidden).contiguous() # [num_layers, batch_size, latent_dim]
        pre_output_vectors = []
        prev_embed = sos_token.expand(batch_size, -1, -1)
        # [batch_size, 256] 
        for i in range(max_len): 
            pre_output, hidden = self.forward_step(prev_embed, hidden)
            pre_output_vectors.append(pre_output)
            if i is max_len - 1:
                break
            token = generator(pre_output.squeeze(1))
            token = torch.max(token, dim=-1)[1]
            embed_token = embeder(token)
            prev_embed = embed_token.unsqueeze(1)
            
        
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return pre_output_vectors

    def generate_v2(self, z, max_len, generator, embeder, sos_token):
        batch_size = z.size(0)
        hidden = self.bridge(z) # z : [batch_size, latent_dim] -> [batch_size, num_layers * hidden_dim]
        hidden = hidden.reshape(batch_size, self.num_layers, self.hidden_dim) #
        hidden = hidden.permute(1, 0, 2) # [num_layers, batch_size, latent_dim]
        hidden = torch.tanh(hidden).contiguous() # [num_layers, batch_size, latent_dim]
        pre_output_vectors = []
        prev_embed = sos_token.expand(batch_size, -1, -1)
        # [batch_size, 256] 
        for i in range(max_len): 
            pre_output, hidden = self.forward_step(prev_embed, hidden)
            pre_output_vectors.append(pre_output)
            if i is max_len - 1:
                break
            token_logits = generator(pre_output.squeeze(1))
            token_probs = torch.softmax(token_logits, dim=-1)
            token = torch.multinomial(token_probs, 1).squeeze(1) # RNN
            embed_token = embeder(token)
            prev_embed = embed_token.unsqueeze(1)
            
        
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return pre_output_vectors
        
        

class VAE(nn.Module):
    def __init__(self,
                 voca_dim,
                 embed_dim,
                 hidden_dim,
                 latent_dim,
                 en_num_layers,
                 de_num_layers,
                 prop_num = 1,
                 run_predictor = True,
                 value_range = None # value_range form : [ '[0,1)' , '[0,10]' ]  
                 ):
        super(VAE, self).__init__()
        # Number of properties

        self.run_predictor = run_predictor
        self.prop_num = prop_num
        self.voca_dim = voca_dim
        self.embed = nn.Embedding(voca_dim, embed_dim)
        self.generator = nn.Sequential(
            nn.Linear(hidden_dim, voca_dim, bias=False),
            nn.LogSoftmax(dim=-1)
        )
        self.encoder = Encoder(embed_dim, hidden_dim, latent_dim, num_layers=en_num_layers) 
        self.decoder = Decoder(embed_dim, hidden_dim, latent_dim, num_layers=de_num_layers)

        if value_range is None:
            self.predictors = nn.ModuleList([
                Predictor(input_dim=latent_dim,
                          output_dim=1)
                for _ in range(prop_num)
                ])
        else:
            self.predictors = nn.ModuleList([
                Predictor(input_dim=latent_dim,
                          output_dim=1,
                          value_range=value_range[i])
                for i in range(prop_num)
                ])

        if not self.run_predictor:
            for param in self.predictors.parameters():
                param.requires_grad = False
        #To Control Property 
        
        self.latent_mask = torch.nn.Parameter(torch.randn(prop_num, latent_dim, 2)) 
        self.tau = 0.5
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def mask_process(self):
        
        odds = torch.sigmoid(self.latent_mask) / (1 - torch.sigmoid(self.latent_mask))
        mask = F.gumbel_softmax(odds, self.tau, hard=True)[:, :, 1] 
        return mask


    def mask_inference(self, thold = 0.5):# self.tau = 0.5 
        odds = torch.sigmoid(self.latent_mask)
        mask = (odds > thold).float()  
        return mask[:, :, 1], odds

    
    def forward(self, src, trg, src_lengths, max_len, tf_ratio = 0.5):
        x = self.embed(src) # batch_size, max_len -> batch_size, max_len -> embedd_dim
        z_mean, z_std = self.encoder(x, src_lengths) #for training process
        latent = self.reparameterize(z_mean, z_std)
        
        y_list = []
        if self.run_predictor:
            latent_mask = self.mask_process().unsqueeze(0) # [1, prop_num, latent_dim]
            y_list = [predictor(latent * latent_mask[:, idx, :]) for idx, predictor in enumerate(self.predictors)]

        reconstruct = self.decoder(trg_embed=self.embed(trg),
                                   z=latent,
                                   max_len=max_len,
                                   generator=self.generator,
                                   embeder=self.embed,
                                   sos_token=self.embed(src[:,0]).unsqueeze(1),
                                   tf_ratio = tf_ratio)
        # [batch, max_len, voca_dim]
        x = self.generator(reconstruct)
        latent_dist_z = (z_mean,z_std)

        if self.run_predictor:
            return x, y_list, latent_dist_z, latent, latent_mask
        else:
            return x, y_list, latent_dist_z, latent

    def inference(self, src, src_lengths, max_len, latent_mask=None):
        x = self.embed(src) # batch_size, max_len -> batch_size, max_len -> embedd_dim
        z_mean, z_std = self.encoder(x, src_lengths) #for training process
        latent = self.reparameterize(z_mean, z_std)
        
        y_list = []
        if self.run_predictor:
            latent_reshape = latent.view(latent.shape[0], 1, -1) #[batch, 1, latent_dim]
            latent_reshape = latent_reshape.repeat(1, self.prop_num, 1) # [batch, prop_num, latent_dim]
            if latent_mask is None:
                #latent_mask = self.mask_inference().unsqueeze( # [1, prop_num, latent_dim]
                latent_mask, _ = self.mask_inference() # [prop_num, latent_dim]
            masking_latent = latent_reshape * latent_mask # Z * J^T   # [batch, prop_num, latent_dim]
            for idx, predictor in enumerate(self.predictors):
                y = predictor(masking_latent[:, idx, :]) #[batch_size, 1] 
                y_list.append(y)

        #(self, z, max_len, generator, embeder, sos_token):
        reconstruct = self.decoder.generate(z=latent,
                                            max_len=max_len,
                                            generator=self.generator,
                                            embeder=self.embed,
                                            sos_token=self.embed(src[:,0]).unsqueeze(1))
        # [batch, max_len, voca_dim]
        x = self.generator(reconstruct)
        latent_dist_z = (z_mean,z_std)

        return x, y_list, latent_dist_z, latent, latent_mask

    def decode_z(self, z, max_len, sos_tensor, return_y = False):
        sos_token = self.embed(sos_tensor)
        reconstruct = self.decoder.generate(z=z,
                                            max_len=max_len,
                                            generator=self.generator,
                                            embeder=self.embed,
                                            sos_token=sos_token.unsqueeze(1))
        x = self.generator(reconstruct)
        latent_mask = self.mask_inference()[0].unsqueeze(0) # [1, prop_num, latent_dim]
        if return_y and self.run_predictor :
            y_list = [predictor(z * latent_mask[:, idx, :]) for idx, predictor in enumerate(self.predictors)]
            return x, y_list
        else:
            return x




