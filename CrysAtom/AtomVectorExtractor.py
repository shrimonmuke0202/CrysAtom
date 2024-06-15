import os
import time
import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import pickle

from data import *
from model import *
from BarlowTwins import *


parser = argparse.ArgumentParser(description='Source code to generate dense vector for atoms')
parser.add_argument('--data-path', type=str, default='./data/',help='Root data path')
parser.add_argument('--disable-cuda', action='store_true',help='Disable CUDA')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=110, type=int, metavar='N',help='number of total epochs to run (default: 110)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,metavar='LR', help='initial learning rate (default: ''0.03)')
parser.add_argument('--dense-vector-dimension', default=50, type=int, metavar='N',help='dimension of the dense representation of atoms')

args=parser.parse_args()

args.cuda = not args.disable_cuda and torch.cuda.is_available()


def main():
    global args
    
    data_path = args.data_path

    dataset = StructureData(data_path)

    datasize = 139308 # Total Number of untagged dataset

    print('Total Datasize :',datasize)

    idx_train=list(range(datasize))
    collate_fn = collate_pool
    train_loader = get_train_loader(dataset=dataset,collate_fn=collate_fn,batch_size=args.batch_size,num_workers=args.workers,pin_memory=args.cuda,train_size=idx_train)


    # Set up of CrysAtom
    structures,_,_,_ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[2].shape[-1]

    model = CrysAtom(orig_atom_fea_len, nbr_fea_len,atom_fea_len=args.dense_vector_dimension,n_conv=5,h_fea_len=128)
    
    if args.cuda:
        model.cuda()

    # SSL loss function: Barlow Twins Loss
    if args.cuda:
        criterion = BarlowTwinsLoss("cuda:0",128,128,0.0051)
    else:
        criterion = BarlowTwinsLoss("cpu",128,128,0.0051)
        
    # Adam Optimizer
    optimizer = optim.Adam(model.parameters(), 0.03 ,weight_decay=0)


    best_loss=999
    best_model = model
    args.start_epoch=0
    loss_train=[]
    adj_loss_train = []
    feat_loss_train = []
    contrastive_loss_train = []

    for epoch in range(args.start_epoch, args.epochs):
        global global_embedding_dict
        global_embedding_dict = {}
        t_epoch = time.time()
        loss, adj_loss, feat_loss, contrastive_losses= train(train_loader, model, criterion, optimizer, epoch)

        if epoch >1:
            loss_train.append(loss)
            adj_loss_train.append(adj_loss)
            feat_loss_train.append(feat_loss)
            contrastive_loss_train.append(contrastive_losses)
        print()
        print(' Epoch Summary : Epoch ' + str(epoch),
              ' Loss: {:.2f}'.format(loss),
              ' Adj Reconst Loss: {:.2f}'.format(adj_loss),
              ' Feat Reconst Loss: {:.2f}'.format(feat_loss),
              '  Barlow Loss: {:.2f}'.format(contrastive_losses),
              ' time: {:.4f} min'.format((time.time() - t_epoch)/60))
        if loss<best_loss:

            file_ = open(os.path.join("./dense_vector/dense_vector_", str(epoch) + ".pkl"),'wb')
            pickle.dump(global_embedding_dict,file_)
            file_.close()

            best_loss=loss

            best_model=model
            print(" Best Loss :"+str(best_loss)+", Saving the model !!")
            torch.save({'state_dict': best_model.state_dict()}, './model/crysatom_state_checkpoint_'+str(epoch)+'.pth.tar')
        
        

# Dense Vector extractor of CrysAtom

def embedding_create(atom_fea,crystal_atom_idx,atom_type):
    embedding_dict = {}
    for idx,i in enumerate(crystal_atom_idx):
        emd_l = atom_fea[i]
        atom_t_l = atom_type[idx]
        for ind,t in enumerate(atom_t_l):
            if t in embedding_dict:
                embedding_dict[t].append(emd_l[ind].view(1,-1).detach().cpu())
            else:
                embedding_dict[t] = []
                embedding_dict[t].append(emd_l[ind].view(1,-1).detach().cpu())
    
    for k in embedding_dict:
        if len(embedding_dict[k])>1:
            embedding_dict[k] = F.normalize(torch.mean(torch.cat(embedding_dict[k],dim=0),dim=0).view(1,-1),dim=1, p=2)
        else:
            embedding_dict[k] = F.normalize(embedding_dict[k][0],dim=1, p=2)
    
    return embedding_dict        
        
# Training procedure and chemical vector extraction.

def train(train_loader, model, criterion, optimizer, epoch):
    losses =[]
    adj_losses = []
    feat_losses = []
    contrastive_losses = []
    # switch to train mode
    model.train()
    for i,(input1,input2,input3,_) in enumerate(tqdm(train_loader)):
        t = time.time()
        atom_fea=input1[0]
        atom_type=input1[1]
        nbr_fea = input1[2]
        nbr_fea_idx = input1[3]
        adj = torch.LongTensor(input1[4])
        crys_index=input1[5]

        if args.cuda:
            input_var = (Variable(atom_fea.cuda(non_blocking=True)),
                         Variable(nbr_fea.cuda(non_blocking=True)),
                         nbr_fea_idx.cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in crys_index])

            input_var_rot_1 = (Variable(input2[0].cuda(non_blocking=True)),
                                    Variable(input2[1].cuda(non_blocking=True)),
                                    input2[2].cuda(non_blocking=True),
                                    [crys_idx.cuda(non_blocking=True) for crys_idx in input2[3]])

            input_var_rot_2 = (Variable(input3[0].cuda(non_blocking=True)),
                    Variable(input3[1].cuda(non_blocking=True)),
                    input3[2].cuda(non_blocking=True),
                    [crys_idx.cuda(non_blocking=True) for crys_idx in input3[3]])
    
        
            atom_fea=atom_fea.cuda()
            adj=adj.cuda()
            
        else:
            input_var = (Variable(atom_fea),
                         Variable(nbr_fea),
                         nbr_fea_idx,
                         crys_index)

            input_var_rot_1 = (Variable(input2[0]),
                               Variable(input2[1]),
                               input2[2],
                               input2[3])

            input_var_rot_2 = (Variable(input3[0]),
                               Variable(input3[1]),
                               input3[2],
                               input3[3])
            

        # compute output
        edge_prob_list, atom_feature_list,crys_fea_list,atom_emd,atom_fea_n = model(*input_var,args.cuda,False) # UL part of CrysAtom
        
        zis = model(*input_var_rot_1,args.cuda,True) # Graph embedding of first agumentation
    
        zjs = model(*input_var_rot_2,args.cuda,True) # Graph embedding of second agumentation

        zis = F.normalize(zis, dim=1)

        zjs = F.normalize(zjs, dim=1)

        embedding_dict = embedding_create(atom_fea_n,input_var[-1],atom_type) # Batch wise Dense Vector Extractor 


        # region Node Level Loss
        # Connection reconstruction
        pos_weight=torch.Tensor([0.1,1,1,1,1,1])
        if args.cuda:
            pos_weight=pos_weight.cuda()
        loss_adj_reconst = F.nll_loss(edge_prob_list, adj,weight=pos_weight)

        # Feature reconstruction
        loss_atom_feat_reconst = F.binary_cross_entropy_with_logits(atom_feature_list, atom_fea)
        # endregion
        
        b_loss = criterion(zis,zjs)  ## SSL part of CrysAtom Framework
        
        for key,item in embedding_dict.items():
            global_embedding_dict[key] = item


        # Final Loss    
        loss = 0.25 * loss_adj_reconst + 0.25 * loss_atom_feat_reconst + 0.50 * b_loss


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        adj_losses.append(loss_adj_reconst.item())
        feat_losses.append(loss_atom_feat_reconst.item())
        contrastive_losses.append(b_loss.item())

        if (i+1)%1000==0:
            print(' Epoch ' + str(epoch),
                  ' Batch ' + str(i),
                  ' Loss: {:.2f}'.format(loss),
                  ' Adj Reconst Loss: {:.2f}'.format(loss_adj_reconst),
                  ' Feat Reconst Loss: {:.2f}'.format(loss_atom_feat_reconst),
                  ' Barlow Loss: {:.2f}'.format(b_loss),
                  ' time: {:.4f} min'.format((time.time() - t) / 60))
    return np.mean(losses),np.mean(adj_losses),np.mean(feat_losses),np.mean(contrastive_losses)


if __name__ == '__main__':
    main()
