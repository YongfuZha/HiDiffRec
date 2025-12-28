
import pickle
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import torch.nn.functional as F
import math
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        # self.relu = nn.ReLU()
        self.gelu=nn.GELU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        # outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = self.dropout2(self.conv2(self.gelu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        # outputs = self.conv2(self.gelu(self.conv1(inputs.transpose(-1, -2))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError

class LightGCN(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        print("num_users:",self.num_users)
        print("num_items:",self.num_items)
        #
        self.users_items_index=self.dataset.users_items_index
        #
        # self.items_users_index=self.dataset.items_users_index
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()

        #
        self.temperature=self.config["temperature"]

        self.max_len=self.get_mean_length()
        self.nhead = self.config["n_head"]
        self.num_encoder_layers = self.config["num_layers"]

        self.hidden_size=self.latent_dim
        self.pos_emb = nn.Embedding(self.max_len, self.hidden_size, padding_idx=0)

        self.emb_dropout = nn.Dropout(p=self.config["emb_dropout"])
        self.dropout=self.config["emb_dropout"]
        self.duibi_dropout=nn.Dropout(p=self.config["duibi_dropout"])

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-8)

        for _ in range(self.num_encoder_layers):
            new_attn_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(self.hidden_size, self.nhead, self.dropout)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.hidden_size, self.dropout)
            self.forward_layers.append(new_fwd_layer)
        print(f"lgn is already to go(dropout:{self.config['dropout']})")
        self.timesteps = 32
        self.betas_start = 0.0001
        self.betas_end = 0.02
        self.latent_dim = self.hidden_size

        self.betas = torch.linspace(self.betas_start, self.betas_end, self.timesteps)
        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=-1)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.denoise_layer2=nn.Sequential(
            nn.Linear(4*self.hidden_size,self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size,self.hidden_size)
        )

        self.denoise_layer3=nn.Sequential(
            nn.Linear(5*self.hidden_size,self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size,self.hidden_size)
        )

        self.w_q = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.init(self.w_q)
        self.w_k = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.init(self.w_k)
        self.w_v = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.init(self.w_v)
        self.ln = torch.nn.LayerNorm(self.hidden_size, elementwise_affine=False)

        # self.begin_epoch=0
        #573*80 396*12 296*100
        # self.end_epoch=573*80
        self.topk=self.config["lofk"]
        self.gamma=self.config["gamma"]
        self.diff_weight=self.config["diff_weight"]

        self.final_mlp=torch.nn.Sequential(
            torch.nn.Linear(4*self.hidden_size,self.hidden_size),
        )

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def init(self,m):
        if isinstance(m,torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias,0)
        elif isinstance(m,torch.nn.Parameter):
            torch.nn.init.xavier_normal_(m)

    def selfAttention(self,features,layer_mask=True):
        features=self.ln(features)
        q=self.w_q(features)
        k=self.w_k(features)
        v=self.w_v(features)

        attn=q.mul(self.hidden_size**-0.5) @ k.transpose(-1,-2)

        if layer_mask:
            mask=torch.ones(5,5).to(world.device)
            mask[3,4]=0
            mask[3,1]=0
            mask[2,0]=0
            mask[2,1]=0
            mask[2,3]=0
            mask[2,4]=0
            mask[1,0]=0
            mask[1,2]=0
            mask[1,3]=0
            mask[1,4]=0
            mask[4,3]=0
            mask[4,1]=0
            attn=attn.masked_fill(mask==0,float('-inf'))
        attn = attn.softmax(dim=-1)
        features=attn @ v
        # print("features.size:",features.size())
        # y=features
        # print("y.size:",y.size())

        return features[:,-2,:],features[:,-1,:],features[:,1:,:]
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                    all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    #消融
    def get_user_emb(self):
        """
                propagate methods for lightGCN
                """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = []
        g_droped=self.Graph
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items

        users_final_emb,_=self.diffusion_denoise(users,users_emb)
        rating = self.f(torch.matmul(users_final_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))


        # diff_users_emb,_=self.get_user_emb()
        diffusion_loss,ssl_loss,users_final_emb,items_diff=self.diffusion_forward(users,users_emb)

        diff_pos_scores = torch.mul(users_final_emb, pos_emb)
        diff_pos_scores = torch.sum(diff_pos_scores, dim=1)
        diff_neg_scores = torch.mul(users_final_emb, neg_emb)
        diff_neg_scores = torch.sum(diff_neg_scores, dim=1)

        bpr_diff_loss = torch.mean(torch.nn.functional.softplus(diff_neg_scores - diff_pos_scores))
        # loss=loss+diff_loss
        # loss=bpr_diff_loss
        return bpr_diff_loss+self.diff_weight*diffusion_loss, reg_loss,ssl_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma

    def layer_computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        g_droped=self.Graph
        users_total=[users_emb]
        items_total=[items_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
            users, items = torch.split(all_emb, [self.num_users, self.num_items])
            users_total.append(users)
            items_total.append(items)

        return users_total,items_total

    def diffusion_forward(self,users,users_emb):
        items_emb=[]
        users_total,items_total=self.layer_computer()
        layer0_emb,layer1_emb,layer2_emb,layer3_emb=users_total[0],users_total[1],users_total[2],users_total[3]
        layer0_items_emb,layer1_items_emb,layer2_items_emb,layer3_items_emb=(items_total[0],items_total[1],
                                                                             items_total[2],items_total[3])
        layer0_users_emb=layer0_emb
        layer1_users_emb=layer1_emb
        layer2_users_emb=layer2_emb
        layer3_users_emb=layer3_emb
        _,neighor_emb = self.get_users_neighor_emb(users.tolist())
        # neighor_emb = self.get_inial_users_neighor_emb(users.tolist())
        #
        times_info = torch.randint(0, self.timesteps, (len(layer2_users_emb),), device=world.device).long()
        times_info_embedding = self.get_time_s(times_info)
        layer1_users_emb_noise = torch.randn_like(layer1_users_emb).to(world.device)
        layer2_users_emb_noise = torch.randn_like(layer2_users_emb).to(world.device)
        layer3_users_emb_noise = torch.randn_like(layer3_users_emb).to(world.device)

        layer2_users_T = self.q_sample(layer2_users_emb, times_info, layer1_users_emb_noise)
        layer3_users_T = self.q_sample(layer3_users_emb, times_info, layer2_users_emb_noise)
        final_users_diff=torch.cat([times_info_embedding[users].unsqueeze(1),layer0_users_emb[users].unsqueeze(1),layer1_users_emb[users].unsqueeze(1),
                                    layer2_users_T[users].unsqueeze(1),layer3_users_T[users].unsqueeze(1)],dim=1)

        #
        # print("final_users_diff.size:",final_users_diff.size())
        l2_pre,l3_pre,final_users_denoise=self.selfAttention(final_users_diff)

        items_emb.append(layer0_items_emb)
        items_emb.append(layer1_items_emb)
        # items_emb.append(layer2_items_diff)
        # items_emb.append(layer3_items_diff)
        items_emb=torch.stack(items_emb,dim=1)
        items_emb=torch.mean(items_emb,dim=1)

        final_concat=torch.cat([layer0_emb[users],layer1_emb[users],l2_pre,l3_pre],dim=-1)
        # final_concat=torch.cat([layer1_emb[users],l2_pre,l3_pre],dim=-1)
        final_emb=self.final_mlp(final_concat)
        user_rep=final_emb
        diffu_loss = F.mse_loss(final_emb, users_emb)
        cl = self.get_duibi(final_emb, neighor_emb)

        return diffu_loss,cl,user_rep,items_emb


    def diffusion_denoise(self, users,users_emb):
        items_emb = []
        users_total, items_total = self.layer_computer()
        layer0_emb, layer1_emb, layer2_emb, layer3_emb = users_total[0], users_total[1], users_total[2], users_total[3]
        layer0_items_emb, layer1_items_emb, layer2_items_emb, layer3_items_emb = (items_total[0], items_total[1],
                                                                                  items_total[2], items_total[3])
        layer0_users_emb = layer0_emb
        layer1_users_emb = layer1_emb
        layer2_users_emb = layer2_emb
        layer3_users_emb = layer3_emb
        # neighor_emb = self.get_users_neighor_emb(users.tolist())
        #

        users_emb_nocon_noise = torch.randn_like(layer2_emb).to(world.device)
        users_emb_con_noise = torch.randn_like(layer3_emb).to(world.device)

        x_target = users_emb_nocon_noise
        x_source = users_emb_con_noise
        layer2_items_diff=layer2_items_emb
        layer3_items_diff=layer3_items_emb
        final_users_denoise=layer1_emb
        for i in reversed(range(0, self.timesteps)):
            t = torch.tensor([i] * x_target.shape[0], dtype=torch.long).to(world.device)
            # -----------------------
            times_info_embeddings = self.get_time_s(t)

            # --
            x_t_target = x_target
            x_t_source=x_source

            final_users_diff = torch.cat(
                [times_info_embeddings.unsqueeze(1),layer0_users_emb.unsqueeze(1),
                 layer1_users_emb.unsqueeze(1),x_target.unsqueeze(1), x_source.unsqueeze(1)], dim=1)
            l2_pre,l3_pre,final_users_denoise = self.selfAttention(final_users_diff)
            model_mean_target = (
                    self.extract(self.posterior_mean_coef1, t, x_t_target.shape) * l2_pre +
                    self.extract(self.posterior_mean_coef2, t, x_t_target.shape) * x_t_target
            )

            model_mean_source = (
                    self.extract(self.posterior_mean_coef1, t, x_t_source.shape) * l3_pre +
                    self.extract(self.posterior_mean_coef2, t, x_t_source.shape) * x_t_source
            )

            if i == 0:
                x_target = model_mean_target
                x_source=model_mean_source
            else:
                # ---
                posterior_variance_t = self.extract(self.posterior_variance, t, x_target.shape)
                noise_target = torch.randn_like(x_target)
                noise_source=torch.randn_like(x_source)

                x_target = model_mean_target + torch.sqrt(posterior_variance_t) * noise_target
                x_source=model_mean_source + torch.sqrt(posterior_variance_t) * noise_source
        items_emb.append(layer0_items_emb)
        items_emb.append(layer1_items_emb)
        items_emb.append(layer2_items_diff)
        items_emb.append(layer3_items_diff)
        items_emb = torch.stack(items_emb, dim=1)
        items_emb = torch.mean(items_emb, dim=1)

        final_concat = torch.cat([layer0_emb[users], layer1_emb[users],x_target[users],x_source[users]], dim=-1)

        users_final_emb = self.final_mlp(final_concat)
        return users_final_emb,items_emb

    def create_tsne_weight(self):
        # zero_user_p=[]
        # first_user_p=[]
        # two_user_p=[]
        # three_user_p=[]
        # final_user_p=[]
        zero_user_p,first_user_p,two_user_p,three_user_p,final_user_p=self.tsne_diffusion_denoise()

        fname="./TSNE/HPUR_Gowalla.pkl"

        print("first_user+p.size:",first_user_p.size())
        dataset={}
        dataset["zero_user_p"]=zero_user_p
        dataset["first_user_p"]=first_user_p
        dataset["two_user_p"]=two_user_p
        dataset["three_user_p"]=three_user_p
        dataset["final_user_p"]=final_user_p
        with open(fname,"wb") as f:
            pickle.dump(dataset,f)
    def tsne_diffusion_denoise(self):
        items_emb = []
        users_total, items_total = self.layer_computer()
        layer0_emb, layer1_emb, layer2_emb, layer3_emb = users_total[0], users_total[1], users_total[2], users_total[3]
        layer0_items_emb, layer1_items_emb, layer2_items_emb, layer3_items_emb = (items_total[0], items_total[1],
                                                                                  items_total[2], items_total[3])
        layer0_users_emb = layer0_emb
        layer1_users_emb = layer1_emb
        layer2_users_emb = layer2_emb
        layer3_users_emb = layer3_emb
        # neighor_emb = self.get_users_neighor_emb(users.tolist())
        #

        users_emb_nocon_noise = torch.randn_like(layer2_emb).to(world.device)
        users_emb_con_noise = torch.randn_like(layer3_emb).to(world.device)

        x_target = users_emb_nocon_noise
        x_source = users_emb_con_noise
        layer2_items_diff=layer2_items_emb
        layer3_items_diff=layer3_items_emb
        final_users_denoise=layer1_emb
        for i in reversed(range(0, self.timesteps)):
            t = torch.tensor([i] * x_target.shape[0], dtype=torch.long).to(world.device)
            # -----------------------
            times_info_embeddings = self.get_time_s(t)

            # --
            x_t_target = x_target
            x_t_source=x_source

            final_users_diff = torch.cat(
                [times_info_embeddings.unsqueeze(1),layer0_users_emb.unsqueeze(1),
                 layer1_users_emb.unsqueeze(1),x_target.unsqueeze(1), x_source.unsqueeze(1)], dim=1)
            l2_pre,l3_pre,final_users_denoise = self.selfAttention(final_users_diff)
            model_mean_target = (
                    self.extract(self.posterior_mean_coef1, t, x_t_target.shape) * l2_pre +
                    self.extract(self.posterior_mean_coef2, t, x_t_target.shape) * x_t_target
            )

            model_mean_source = (
                    self.extract(self.posterior_mean_coef1, t, x_t_source.shape) * l3_pre +
                    self.extract(self.posterior_mean_coef2, t, x_t_source.shape) * x_t_source
            )

            if i == 0:
                x_target = model_mean_target
                x_source=model_mean_source
            else:
                # ---
                posterior_variance_t = self.extract(self.posterior_variance, t, x_target.shape)
                noise_target = torch.randn_like(x_target)
                noise_source=torch.randn_like(x_source)

                x_target = model_mean_target + torch.sqrt(posterior_variance_t) * noise_target
                x_source=model_mean_source + torch.sqrt(posterior_variance_t) * noise_source
        items_emb.append(layer0_items_emb)
        items_emb.append(layer1_items_emb)
        items_emb.append(layer2_items_diff)
        items_emb.append(layer3_items_diff)
        items_emb = torch.stack(items_emb, dim=1)
        items_emb = torch.mean(items_emb, dim=1)
        final_concat = torch.cat([layer0_emb, layer1_emb,x_target,x_source], dim=-1)
        # final_concat = torch.cat([final_users_denoise[:,0,:][users], final_users_denoise[:,1,:][users],
        #                           final_users_denoise[:,2,:][users],final_users_denoise[:,3,:][users]], dim=-1)
        users_final_emb = self.final_mlp(final_concat)
        return layer0_emb,layer1_emb,x_target,x_source,users_final_emb
    def get_inial_users_neighor_emb(self,users_data):

        # #
        neighor_seqs = []
        for user in users_data:
            neighor_seqs.append(self.get_user_padding_items(user))
            # print(":",self.get_user_padding_items(user))
        # neighor_emb = self.log2feats(np.array(neighor_seqs))
        neighor_seqs = torch.LongTensor(neighor_seqs).to(world.device)
        neighor_emb = self.embedding_item.weight[neighor_seqs]
        neighor_emb = torch.mean(neighor_emb, dim=1)
        neighor_emb = neighor_emb.view(len(neighor_emb), -1)
        return neighor_emb


    def get_users_neighor_emb(self, users_data):
        #
        neighor_seqs = []
        for user in users_data:
            padding_result = self.get_user_padding_items(user)
            #
            if isinstance(padding_result, tuple):
                padding_result = padding_result[0]
            neighor_seqs.append(padding_result)

        #
        neighor_seqs = torch.LongTensor(neighor_seqs).to(world.device)
        neighor_emb = self.embedding_item.weight[neighor_seqs]
        # print("neighor_emb.size:", neighor_emb.size())

        k = self.topk
        lrd_result = self.calculate_lrd(neighor_emb, k)
        #
        lrd = lrd_result[0] if isinstance(lrd_result, tuple) else lrd_result
        # print("lrd.size:", lrd.size())

        #  LOF
        lof_result = self.calculate_lof(neighor_emb, lrd, k)
        lof = lof_result[0] if isinstance(lof_result, tuple) else lof_result
        # print("lof.size:", lof.size())

        #
        clean_items_mask = self.get_clean_items_mask(neighor_emb, lof, gamma=self.gamma)
        # print("clean_items_mask.size:", clean_items_mask.size())

        #
        user_representation = self.calculate_user_representation(neighor_emb, clean_items_mask)
        # print("user_representataion.size:", user_representation.size())

        return neighor_seqs, user_representation

    def calculate_lrd(self, E, k=5):

        #
        if isinstance(E, tuple):
            E = E[0]

        batch_size, length, hidden_size = E.shape

        #
        k = min(k, length - 1)
        k = max(1, k)

        #
        dist_matrix = torch.cdist(E, E, p=2)  # [batch_size, length, length]

        #
        diagonal_mask = torch.eye(length, device=E.device).unsqueeze(0).expand(batch_size, -1, -1)
        dist_matrix_masked = dist_matrix + diagonal_mask * 1e6

        #
        try:
            topk_result = dist_matrix_masked.topk(k, largest=False, dim=2)
            #
            nearest_neighbors_dist = topk_result.values  # [batch_size, length, k]
            nearest_neighbors_idx = topk_result.indices  # [batch_size, length, k]
        except AttributeError:
            #
            topk_result = dist_matrix_masked.topk(k, largest=False, dim=2)
            nearest_neighbors_dist = topk_result[0]  # [batch_size, length, k]
            nearest_neighbors_idx = topk_result[1]  # [batch_size, length, k]

        #
        k_distances = nearest_neighbors_dist[:, :, -1]  # [batch_size, length]

        batch_idx = torch.arange(batch_size, device=E.device).view(-1, 1, 1).expand(-1, length, k)
        neighbors_k_distances = k_distances[batch_idx, nearest_neighbors_idx]  # [batch_size, length, k]


        reach_distances = torch.max(nearest_neighbors_dist, neighbors_k_distances)  # [batch_size, length, k]


        avg_reach_distance = reach_distances.mean(dim=2)  # [batch_size, length]
        lrd = 1.0 / (avg_reach_distance + 1e-8)  # [batch_size, length]

        return lrd

    def calculate_lof(self, E, lrd, k=5):

        if isinstance(E, tuple):
            E = E[0]
        if isinstance(lrd, tuple):
            lrd = lrd[0]

        batch_size, length, hidden_size = E.shape

        #
        k = min(k, length - 1)
        k = max(1, k)

        #
        dist_matrix = torch.cdist(E, E, p=2)  # [batch_size, length, length]

        #
        diagonal_mask = torch.eye(length, device=E.device).unsqueeze(0).expand(batch_size, -1, -1)
        dist_matrix = dist_matrix + diagonal_mask * 1e6

        #
        try:
            topk_result = dist_matrix.topk(k, largest=False, dim=2)
            nearest_neighbors_idx = topk_result.indices  # [batch_size, length, k]
        except AttributeError:
            topk_result = dist_matrix.topk(k, largest=False, dim=2)
            nearest_neighbors_idx = topk_result[1]  # [batch_size, length, k]

        #
        #
        if isinstance(lrd, tuple):
            lrd = lrd[0]

        batch_idx = torch.arange(batch_size, device=E.device).view(-1, 1, 1).expand(-1, length, k)
        nearest_neighbors_lrd = lrd[batch_idx, nearest_neighbors_idx]  # [batch_size, length, k]

        #
        lrd_expanded = lrd.unsqueeze(2).expand(-1, -1, k)  # [batch_size, length, k]
        lrd_ratio = nearest_neighbors_lrd / (lrd_expanded + 1e-8)  # [batch_size, length, k]
        lof = lrd_ratio.mean(dim=2)  # [batch_size, length]

        #
        return lof

    def get_clean_items_mask(self, E, lof, gamma=1.5):

        batch_size, length = lof.shape

        #
        clean_mask = lof <= gamma  # [batch_size, length]

        #
        has_clean_items = clean_mask.sum(dim=1) > 0  # [batch_size]

        #
        argmin_result = torch.argmin(lof, dim=1)  # [batch_size]
        min_lof_idx = argmin_result
        batch_idx = torch.arange(batch_size, device=lof.device)

        #
        min_lof_mask = torch.zeros_like(clean_mask)
        min_lof_mask[batch_idx, min_lof_idx] = True

        #
        final_mask = torch.where(has_clean_items.unsqueeze(1), clean_mask, min_lof_mask)

        return final_mask

    def calculate_user_representation(self, E, clean_mask):

        batch_size, length, hidden_size = E.shape

        #
        mask_expanded = clean_mask.unsqueeze(2).expand(-1, -1, hidden_size)  # [batch_size, length, hidden_size]

        #
        masked_embeddings = E * mask_expanded.float()  # [batch_size, length, hidden_size]

        #
        clean_counts = clean_mask.sum(dim=1, keepdim=True).float()  # [batch_size, 1]

        #
        user_representation = masked_embeddings.sum(dim=1) / clean_counts  # [batch_size, hidden_size]

        return user_representation


    #
    def get_duibi(self, users_emb,neighor_emb):

        #
        #----------------------------------------
        g_users_emb = users_emb


        # neighor_emb, _ = self.gcn_duibi_computer()
        user_emb1 = g_users_emb
        user_emb2 = neighor_emb
        normalize_user_emb1 = torch.norm(user_emb1, p=2, dim=1)
        normalize_user_emb2 = torch.norm(user_emb2, p=2, dim=1)
        # normalize_n_users_emb = torch.norm(n_users_emb, p=2, dim=1)

        normalize_user_emb1 = normalize_user_emb1.view(len(normalize_user_emb1), -1)
        normalize_user_emb2 = normalize_user_emb2.view(len(normalize_user_emb2), -1)
        # normalize_n_users_emb = normalize_n_users_emb.view(len(normalize_n_users_emb), -1)

        final_user_emb1 = user_emb1 / normalize_user_emb1
        final_user_emb2 = user_emb2 / normalize_user_emb2
        # final_user_emb3 = n_users_emb / normalize_n_users_emb

        pos_score_user = torch.sum(torch.multiply(final_user_emb1, final_user_emb2), dim=1)
        ttl_score_user = torch.matmul(final_user_emb1, final_user_emb2.T) - torch.diag(pos_score_user)
        # ttl_score_user = torch.matmul(final_user_emb1, final_user_emb2.T)

        pos_score_user = torch.exp(pos_score_user / self.temperature)
        ttl_score_user = torch.sum(torch.exp(ttl_score_user / self.temperature), dim=1)

        ssl_loss_user = -torch.sum(torch.log(pos_score_user / ttl_score_user))

        ssl_loss = ssl_loss_user/(len(users_emb))

        return ssl_loss


    #
    def get_user_padding_items(self, user):
        if user == 0:
            return [0]*self.max_len
        neighor=self.users_items_index[user]
        neighor_length=len(neighor)

        if neighor_length < self.max_len:
            neighor = np.array(neighor)
            neighor=np.random.choice(neighor,self.max_len).tolist()
        else:
            neighor = np.array(neighor)
            neighor = np.random.choice(neighor, self.max_len, replace=False).tolist()
        return neighor

    #
    def get_item_padding_users(self,item):
        neighor=np.random.choice(self.items_users_index[item],self.max_len).tolist()
        if len(neighor)<self.max_len:
            neighor=[0]*(self.max_len-len(neighor))+neighor

        return neighor

    #
    def duibi_computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        # torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                # all_emb = (1.0/(layer+2))*torch.sparse.mm(g_droped, all_emb)
                all_emb = torch.sparse.mm(g_droped, all_emb)
            all_emb=self.duibi_dropout(all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        #
        # light_out=self.duibi_dropout(light_out)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    def gcn_duibi_computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        # torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]

        g_droped=self.Graph
        for layer in range(1):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                # all_emb = (1.0/(layer+2))*torch.sparse.mm(g_droped, all_emb)
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        # embs = torch.stack(embs, dim=1)
        light_out=embs[1]
        # light_out = torch.mean(embs, dim=1)
        #
        # light_out=self.duibi_dropout(light_out)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    #
    def get_mean_length(self):
        mean_length=0
        total_users=0
        for user,items in self.users_items_index.items():
            mean_length+=len(items)
            total_users+=1
        return int(mean_length/total_users)

    #
    def get_items_users_mean_length(self):
        mean_length=0
        total_items=0
        for item,users in self.items_users_index.items():
            mean_length+=len(users)
            total_items+=1

        return int(mean_length/total_items)

    def get_time_s(self, time):
        half_dim = self.latent_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=world.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(world.device)

    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):

        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    def q_sample(self, x_start, t, noise=None):
        # print(self.betas)
        if noise is None:
            noise = torch.randn_like(x_start)
            # noise = torch.randn_like(x_start) / 100
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        # mean=torch.mean(x_start,dim=0)
        # noise=mean+(x_start - mean).pow(2).mean().sqrt()*noise
        # return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise*torch.sign(x_start)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, neighor_emb, noise=None, loss_type="l2"):
        #
        if noise is None:
            noise = torch.randn_like(x_start)
            # noise = torch.randn_like(x_start) / 100

        #
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        t = self.get_time_s(t)
        neighor_emb = neighor_emb.view(neighor_emb.shape[0], -1)
        predicted_x = self.diffuer_model(torch.cat([x_noisy, neighor_emb, t], dim=-1))

        #
        if loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x)
        elif loss_type == 'l2':
            loss = F.mse_loss(x_start, predicted_x)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x)
        else:
            raise NotImplementedError()

        return loss, predicted_x

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    @torch.no_grad()
    def p_sample(self,x, t, t_index,diffuer_mlp):
        times_info = torch.tensor([t_index] * x.shape[0], dtype=torch.long).to(world.device)
        times_info_embeddings = self.get_time_s(times_info)
        x_start = diffuer_mlp(torch.cat([x, times_info_embeddings],dim=-1))
        x_t = x
        model_mean = (
                self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def denoise(self, times_info,x_T,diffuer_mlp):
        # x = self.q_sample(x, times_info)
        # x=torch.randn_like(condtion_info)
        for i in reversed(range(0, self.timesteps)):
            x_T = self.p_sample(x_T, torch.full((times_info.shape[0],), i, device=world.device, dtype=torch.long), i,diffuer_mlp)
        return x_T
