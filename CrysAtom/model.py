import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat([atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out
    
    

class CrysAtom(nn.Module):
    """
    Create a Deep GNN based Encoder Decoder Model for Crystalline Materials to learn
    representation in an unsupervised way.
    """

    def __init__(self, orig_atom_fea_len, nbr_fea_len, atom_fea_len=64, n_conv=3,h_fea_len=128):
        super(CrysAtom, self).__init__()

        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len, bias=False)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])


        self.fc_adj = nn.Bilinear(atom_fea_len, atom_fea_len, 6)
        self.fc1 = nn.Linear(6, 6)

        self.fc_atom_feature = nn.Linear(atom_fea_len, orig_atom_fea_len)
        
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        self.fc_head = nn.Sequential(
            nn.Linear(h_fea_len, h_fea_len), 
            nn.Softplus(),
            nn.Linear(h_fea_len, h_fea_len)
        )

    def forward(self,atom_fea,nbr_fea, nbr_fea_idx,crystal_atom_idx,cuda_flag,loss_flag=False):
        # Encoder Part (Crystal Graph Convolution Encoder)
        atom_fea = self.embedding(atom_fea)  
        
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        
        # Graph Embedding of augmented crystal obtained from same crystal, which helps to learn Deep GNN in self supervised way
        if loss_flag:
            crys_fea = self.pooling(atom_fea, crystal_atom_idx)
            crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
            crys_fea = self.conv_to_fc_softplus(crys_fea)
            return self.fc_head(crys_fea)
        else:
            atom_emb = []

            bt_atom_fea = [atom_fea[idx_map] for idx_map in crystal_atom_idx]

            edge_prob_list = []
            atom_feature_list = []
            sg_pred_list=[]
            crys_fea_list=[]
            for i in range(len(bt_atom_fea)):
                atom_fea_=bt_atom_fea[i]
                atom_fea_ = F.normalize(atom_fea_, dim=1, p=2)
                atom_emb.append(atom_fea_)
                z_G = torch.mean(atom_fea_, dim=0, keepdim=True)
                crys_fea_list.append(z_G)
                N = atom_fea_.shape[0]
                dim = atom_fea_.shape[1]

                # Repeat feature N times : (N,N,dim)
                atom_nbr_fea = atom_fea_.repeat(N, 1, 1)
                atom_nbr_fea = atom_nbr_fea.contiguous().view(-1, dim)

                # Expand N times : (N,N,dim)
                atom_adj_fea = torch.unsqueeze(atom_fea_, 1).expand(N, N, dim)
                atom_adj_fea = atom_adj_fea.contiguous().view(-1, dim)

                # Decoder

                # Bilinear Layer : Adjacency List Reconstruction
                edge_p = self.fc_adj(atom_adj_fea, atom_nbr_fea)
                edge_p = self.fc1(edge_p)
                edge_p=F.log_softmax(edge_p, dim=1)
                edge_prob_list.append(edge_p)

                # Node Feature Reconstruction
                atom_feature_list.append(self.fc_atom_feature(atom_fea_))

            atom_feature_list = torch.cat(atom_feature_list, dim=0)

            edge_prob_list = torch.cat(edge_prob_list, dim=0)

            crys_fea_list = torch.cat(crys_fea_list, dim=0)
            return edge_prob_list, atom_feature_list,crys_fea_list,atom_emb,atom_fea

    
    def pooling(self, atom_fea, crystal_atom_idx):
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)