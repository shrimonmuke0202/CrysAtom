import csv
import functools
import json
import os
import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import math
import pickle


def get_train_loader(dataset, collate_fn=default_collate,batch_size=64,num_workers=4, pin_memory=False, **kwargs):

    train_size = kwargs['train_size']
    train_sampler = SubsetRandomSampler(train_size)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)


    return train_loader



class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)
    
class AtomInitializer(object):
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]
    


class AtomCustomJSONInitializer(AtomInitializer):
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)
            

def get_molecular_adj(nbr_fea_idx,N):
    edge_list = nbr_fea_idx
    adj = np.zeros((N, N),dtype=int)
    for i in range(N):
        for j in edge_list[i]:
            if adj[i][j]<=4:
                adj[i][j]=adj[i][j]+1
    return adj



def collate_pool(dataset_list):
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    batch_adj, batch_adj_ind,batch_node_label,batch_sg,batch_sg_no = [], [], [],[], []
    
    batch_atom_fea_rot_1, batch_nbr_fea_rot_1, batch_nbr_fea_idx_rot_1 = [], [], []
    batch_atom_fea_rot_2, batch_nbr_fea_rot_2, batch_nbr_fea_idx_rot_2 = [], [], []
    
    crystal_atom_idx, batch_target = [], []
    batch_atom_type = []

    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, atom_type ,nbr_fea, nbr_fea_idx,adj),( atom_fea_rot_1 , nbr_fea_rot_1 , nbr_fea_idx_rot_1 ), ( atom_fea_rot_2 , nbr_fea_rot_2 , nbr_fea_idx_rot_2 ), cif_id) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_atom_type.append(atom_type)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)

        batch_adj.append(adj.tolist())
        
        batch_atom_fea_rot_1.append(atom_fea_rot_1)
        batch_atom_fea_rot_2.append(atom_fea_rot_2)
        batch_nbr_fea_rot_1.append(nbr_fea_rot_1)
        batch_nbr_fea_rot_2.append(nbr_fea_rot_2)
        batch_nbr_fea_idx_rot_1.append(nbr_fea_idx_rot_1+base_idx)
        batch_nbr_fea_idx_rot_2.append(nbr_fea_idx_rot_2+base_idx)

        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            batch_atom_type,
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            np.concatenate(batch_adj),

            crystal_atom_idx),\
            (torch.cat(batch_atom_fea_rot_1, dim=0),
            torch.cat(batch_nbr_fea_rot_1, dim=0),
            torch.cat(batch_nbr_fea_idx_rot_1, dim=0),
            crystal_atom_idx), \
    		(torch.cat(batch_atom_fea_rot_2, dim=0),
            torch.cat(batch_nbr_fea_rot_2, dim=0),
            torch.cat(batch_nbr_fea_idx_rot_2, dim=0),
            crystal_atom_idx),\
        batch_cif_ids



class StructureData(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.max_num_nbr=12
        self.radius = 8
        self.dmin = 0
        self.step = 0.2
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        atom_init_file = os.path.join('./atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=self.dmin, dmax=self.radius, step=self.step)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    
    def __len__(self):
        file_lst = glob.glob(self.root_dir + "/*.pkl")
        return len(file_lst)
        
    
    def __getitem__(self, idx):
        cif_id = str(idx)
        
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id + '.cif'))

        atom_type = []
        atom_fea_list = []
        for i in range(len(crystal)):
            atom_type.append(crystal[i].specie.number)
            atom_fea_list.append(self.ari.get_atom_fea(crystal[i].specie.number))
            
        atom_fea = np.vstack(atom_fea_list)
        
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
                
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        adj = get_molecular_adj(nbr_fea_idx, nbr_fea.shape[0])
        adj = adj.flatten()

        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        adj = torch.LongTensor(adj)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        
        (atom_fea_rot_1, nbr_fea_rot_1, nbr_fea_idx_rot_1), (atom_fea_rot_2, nbr_fea_rot_2, nbr_fea_idx_rot_2) = self.create_augment(crystal)
        
        return (atom_fea,atom_type,nbr_fea, nbr_fea_idx, adj),(atom_fea_rot_1, nbr_fea_rot_1, nbr_fea_idx_rot_1), (atom_fea_rot_2, nbr_fea_rot_2, nbr_fea_idx_rot_2), cif_id
    
    def create_augment(self,crys):
        crystal_rot_1 = crys.copy()
        crystal_rot_2 = crys.copy()
        
        num_sites = crys.num_sites
        
        
        mask_num = int(max([1, math.floor(0.10*num_sites)]))
        indices_remove_1 = np.random.choice(num_sites, mask_num, replace=False)
        indices_remove_2 = np.random.choice(num_sites, mask_num, replace=False)
        
        atom_fea_rot_1 = np.vstack([self.ari.get_atom_fea(crystal_rot_1[i].specie.number)
                              for  i  in  range ( len ( crystal_rot_1 ))])
        
        # mask 10% atoms
        atom_fea_rot_1[indices_remove_1,:] = 0
        
        atom_fea_rot_1 = torch.Tensor(atom_fea_rot_1)
        all_nbrs_rot_1 = crystal_rot_1.get_all_neighbors(self.radius, include_index=True)
        all_nbrs_rot_1 = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs_rot_1]
        nbr_fea_idx_rot_1 , nbr_fea_rot_1  = [], []
        
        for nbr in all_nbrs_rot_1:
            if len(nbr) < self.max_num_nbr:
                
                # mask 10% edges
                keep_edge_num = int(np.ceil(0.90*len(nbr)))
                keep_edge_indices = np.random.choice(len(nbr), keep_edge_num, replace=False)
                nbr  = [ nbr [ i ] for  i  in  keep_edge_indices ]
                nbr_fea_idx_rot_1.append(list(map(lambda x: x[2], nbr)) +
                                  [0] * (self.max_num_nbr - keep_edge_num))
                nbr_fea_rot_1.append(list(map(lambda x: x[1], nbr)) +
                              [self.radius + 1.] * (self.max_num_nbr - keep_edge_num))
            
            else:
                # mask 10% edges
                keep_edge_num = int(np.ceil(0.90*self.max_num_nbr))
                keep_edge_indices = np.random.choice(self.max_num_nbr, keep_edge_num, replace=False)
                nbr  = [ nbr [ i ] for  i  in  keep_edge_indices ]
                nbr_fea_idx_rot_1.append(list(map(lambda x: x[2], nbr)) +
                                  [0] * (self.max_num_nbr - keep_edge_num))
                nbr_fea_rot_1.append(list(map(lambda x: x[1], nbr)) +
                              [self.radius + 1.] * (self.max_num_nbr - keep_edge_num))
                
        nbr_fea_idx_rot_1, nbr_fea_rot_1 = np.array(nbr_fea_idx_rot_1), np.array(nbr_fea_rot_1)
        nbr_fea_rot_1 = self.gdf.expand(nbr_fea_rot_1)
        atom_fea_rot_1 = torch.Tensor(atom_fea_rot_1)
        nbr_fea_rot_1 = torch.Tensor(nbr_fea_rot_1)
        nbr_fea_idx_rot_1 = torch.LongTensor(nbr_fea_idx_rot_1)
        
        atom_fea_rot_2 = np.vstack([self.ari.get_atom_fea(crystal_rot_2[i].specie.number)
                              for  i  in  range ( len ( crystal_rot_2 ))])
        
        # mask 10% atoms
        atom_fea_rot_2[indices_remove_2,:] = 0

        atom_fea_rot_2 = torch.Tensor(atom_fea_rot_2)
        all_nbrs_rot_2 = crystal_rot_2.get_all_neighbors(self.radius, include_index=True)
        all_nbrs_rot_2 = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs_rot_2]
        nbr_fea_idx_rot_2 , nbr_fea_rot_2  = [], []
        
        for nbr in all_nbrs_rot_2:
            if len(nbr) < self.max_num_nbr:
            
                # mask 10% edges
                keep_edge_num = int(np.ceil(0.90*len(nbr)))
                keep_edge_indices = np.random.choice(len(nbr), keep_edge_num, replace=False)
                nbr  = [ nbr [ i ] for  i  in  keep_edge_indices ]
                nbr_fea_idx_rot_2.append(list(map(lambda x: x[2], nbr)) +
                                  [0] * (self.max_num_nbr - keep_edge_num))
                nbr_fea_rot_2.append(list(map(lambda x: x[1], nbr)) +
                              [self.radius + 1.] * (self.max_num_nbr - keep_edge_num))

            
            else:
                # mask 10% edges
                keep_edge_num = int(np.ceil(0.90*self.max_num_nbr))
                keep_edge_indices = np.random.choice(self.max_num_nbr, keep_edge_num, replace=False)
                nbr  = [ nbr [ i ] for  i  in  keep_edge_indices ]
                nbr_fea_idx_rot_2.append(list(map(lambda x: x[2], nbr)) +
                                  [0] * (self.max_num_nbr - keep_edge_num))
                nbr_fea_rot_2.append(list(map(lambda x: x[1], nbr)) +
                              [self.radius + 1.] * (self.max_num_nbr - keep_edge_num))
        
        nbr_fea_idx_rot_2, nbr_fea_rot_2 = np.array(nbr_fea_idx_rot_2), np.array(nbr_fea_rot_2)
        nbr_fea_rot_2 = self.gdf.expand(nbr_fea_rot_2)
        atom_fea_rot_2 = torch.Tensor(atom_fea_rot_2)
        nbr_fea_rot_2 = torch.Tensor(nbr_fea_rot_2)
        nbr_fea_idx_rot_2 = torch.LongTensor(nbr_fea_idx_rot_2)
        return (atom_fea_rot_1, nbr_fea_rot_1, nbr_fea_idx_rot_1), (atom_fea_rot_2, nbr_fea_rot_2, nbr_fea_idx_rot_2)
