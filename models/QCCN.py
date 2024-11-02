import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import ResNet

class QCCN(nn.Module):

    def  __init__(self, way=None, shots=None, resnet=False, resnet18=False, is_pretraining=False, num_cat=None):

        super().__init__()
        self.resolution = 5 * 5
        if resnet:
            self.num_channel = 640
            self.feature_extractor = ResNet.resnet12()
        elif resnet18:
            self.num_channel = 640
            self.feature_extractor = ResNet.resnet18()

        self.shots = shots
        self.way = way
        self.resnet = resnet
        self.resnet18 = resnet18

        # number of channels for the feature map, correspond to d in the paper
        self.d = self.num_channel

        # temperature scaling, correspond to gamma in the paper
        self.scale = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)
        self.scale2 = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        # H*W=5*5=25, resolution of feature map, correspond to r in the paper

        self.r = nn.Parameter(torch.zeros(2),requires_grad=not is_pretraining)
        self.r2 = nn.Parameter(torch.zeros(2), requires_grad=not is_pretraining)
        self.r3 = nn.Parameter(torch.zeros(2),requires_grad=not is_pretraining)
        self.r4 = nn.Parameter(torch.zeros(2), requires_grad=not is_pretraining)

        if is_pretraining:
            # number of categories during pre-training
            self.num_cat = num_cat
            # category matrix, correspond to matrix M of section 3.6 in the paper
            self.cat_mat = nn.Parameter(torch.randn(self.num_cat,self.resolution,self.d),requires_grad=True)


    def get_feature_map(self,inp):

        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp) 

        if self.resnet or self.resnet18:
            feature_map = feature_map / np.sqrt(640)
        return feature_map.view(batch_size, self.num_channel, -1).permute(0, 2, 1).contiguous()  # N,HW,C

    def get_recon_dist(self, query, support, alpha, beta, Woodbury=True):
        # query: way*query_shot*resolution, d
        # support: way, shot*resolution , d
        # Woodbury: whether to use the Woodbury Identity as the implementation or not
        # correspond to kr/d in the paper
        reg = support.size(1) / support.size(2)

        # correspond to lambda in the paper
        lam = reg * alpha.exp() + 1e-6

        # correspond to gamma in the paper
        rho = beta.exp()

        st = support.permute(0, 2, 1)  # way, d, shot*resolution

        if Woodbury:
            # correspond to Equation 10 in the paper
            sts = st.matmul(support)  # way, d, d
            m_inv = (sts + torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse()  # way, d, d
            hat = m_inv.matmul(sts)  # way, d, d
        else:
            # correspond to Equation 8 in the paper
            sst = support.matmul(st)  # way, shot*resolution, shot*resolution
            m_inv = (sst + torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lam)).inverse()  # way, shot*resolution, shot*resolutionsf
            hat = st.matmul(m_inv).matmul(support)  # way, d, d

        Q_bar = query.matmul(hat).mul(rho)  # way, way*query_shot*resolution, d
        dist = (Q_bar - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0)  # way*query_shot*resolution, way

        return dist

    def get_reconS_dist(self, support, support_pool, alpha, beta, Woodbury=True):
        reg = support_pool.size(1) / support_pool.size(2)

        lam = reg * alpha.exp() + 1e-6

        rho = beta.exp()

        st = support_pool.permute(0, 2, 1)  # way, d, shot*resolution
        if Woodbury:
            sts = st.matmul(support_pool)  # way, d, d
            m_inv = (sts + torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse()  # way, d, d
            hat = m_inv.matmul(sts)  # way, d, d
        else:
      
            sst = support_pool.matmul(st)  # way, shot*resolution, shot*resolution
            m_inv = (sst + torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lam)).inverse()  # way, shot*resolution, shot*resolutionsf
            hat = st.matmul(m_inv).matmul(support_pool)  # way, d, d

        S_bar = support.matmul(hat).mul(rho)  # way, way*query_shot*resolution, d
        dist = (S_bar - support.unsqueeze(0)).pow(2).sum(2).permute(1, 0)  # way*query_shot*resolution, way
        return dist


    def get_neg_l2_dist(self,inp,inp_da,way,shot,query_shot,return_support=False):

        resolution = self.resolution
        d = self.num_channel
        alpha = self.r[0]
        beta = self.r[1]

        alpha2 = self.r2[0]
        beta2 = self.r2[1]

        alpha3 = self.r3[0]
        beta3 = self.r3[1]

        alpha4 = self.r4[0]
        beta4 = self.r4[1]

        feature_map = self.get_feature_map(inp)  # N,HW,C


        # S-->Q 
        support = feature_map[:way * shot].view(way, shot * resolution, d)  
        query = feature_map[way * shot:].view(way * query_shot * resolution, d)  
        recon_dist = self.get_recon_dist(query=query, support=support, alpha=alpha,beta=beta)  # way*query_shot*resolution, way
        dist = recon_dist.neg().view(way * query_shot, resolution, way).mean(1)  # way*query_shot, way

        if return_support:
            # S--->S
            support_view = feature_map[:way * shot].view(way * shot * resolution, d)
            recon_dist4 = self.get_reconS_dist(support=support_view, support_pool=support, alpha=alpha4, beta=beta4)
            dist4 = recon_dist4.neg().view(way * shot, resolution, way).mean(1)

            feature_da = self.get_feature_map(inp_da)  # N,HW,C 
            # (S+)-->Q 
            support2 = feature_da[:way * shot].view(way, shot * resolution, d)  
            recon_dist2 = self.get_recon_dist(query=query, support=support2, alpha=alpha2,beta=beta2)  # way*query_shot*resolution, way
            dist2 = recon_dist2.neg().view(way * query_shot, resolution, way).mean(1)  # way*query_shot, way

            # (S+)--->S
            recon_dist3 = self.get_reconS_dist(support=support_view, support_pool=support2, alpha=alpha3, beta=beta3)
            dist3 = recon_dist3.neg().view(way * shot, resolution, way).mean(1)
            dist = (dist + dist2) / 2
            dist_s = (dist3 + dist4) / 2
            return dist, dist_s, support
        else:
            return dist


    def meta_test(self,inp,way,shot,query_shot):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,inp_da='',
                                           way=way,
                                           shot=shot,
                                           query_shot=query_shot)
        _,max_index = torch.max(neg_l2_dist,1)
        return max_index


    def forward_pretrain(self,inp):

        feature_map = self.get_feature_map(inp) 
        batch_size = feature_map.size(0)

        feature_map = feature_map.view(batch_size*self.resolution,self.d).contiguous()

        alpha = self.r[0]
        beta = self.r[1]
        recon_dist = self.get_recon_dist(query=feature_map,support=self.cat_mat,alpha=alpha,beta=beta) # way*query_shot*resolution, way

        neg_l2_dist = recon_dist.neg().view(batch_size,self.resolution,self.num_cat).mean(1) # batch_size,num_cat

        logits = neg_l2_dist * self.scale
        log_prediction = F.log_softmax(logits, dim=1)

        return log_prediction


    def forward(self,inp,support_da):
        neg_l2_dist, dist_s,support = self.get_neg_l2_dist(inp=inp, inp_da=support_da,
                                                    way=self.way,
                                                    shot=self.shots[0],
                                                    query_shot=self.shots[1],
                                                    return_support=True)

        logits = neg_l2_dist * self.scale
        log_prediction = F.log_softmax(logits, dim=1)

        logits_s = dist_s * self.scale2
        log_prediction_s = F.log_softmax(logits_s, dim=1)

        return log_prediction, log_prediction_s,support
