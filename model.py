import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from copy import deepcopy


class ShotModel(nn.Module):
    def __init__(self, config, utils):
        super(ShotModel, self).__init__()

        self.embedding_dims = config['embedding_dims']
        self.w_embedding_dim = self.embedding_dims['W']
        self.period_embedding_dim = self.embedding_dims['PERIOD']
        self.pts_type_embedding_dim = self.embedding_dims['PTS_TYPE']
        self.defender_embedding_dim = self.embedding_dims['CLOSEST_DEFENDER_PLAYER_ID']
        self.player_embedding_dim = self.embedding_dims['player_id']

        self.w_embedding = nn.Embedding(len(utils.w_map), self.w_embedding_dim)
        self.period_embedding = nn.Embedding(len(utils.period_map), self.period_embedding_dim)
        self.pts_type_embedding = nn.Embedding(len(utils.pts_type_map), self.pts_type_embedding_dim)
        self.defender_embedding = nn.Embedding(len(utils.defender_map), self.defender_embedding_dim)
        self.player_embedding = nn.Embedding(len(utils.player_map), self.player_embedding_dim)

        self.cont_layer_dims = config['cont_layers']
        self.cont_layer_dims.insert(0, len(utils.cont_cols))
        self.cont_layers = []
        for idx in np.arange(len(self.cont_layer_dims)-1):
            in_features = self.cont_layer_dims[idx]
            out_features = self.cont_layer_dims[idx+1]
            self.cont_layers.append(nn.Linear(in_features, out_features))
            self.cont_layers.append(nn.ReLU())
        self.cont_forward = nn.Sequential(*self.cont_layers)

        self.cat_layer_dims = config['cat_layers']
        self.cat_layer_dims.insert(0, self.w_embedding_dim+self.period_embedding_dim+self.pts_type_embedding_dim+self.defender_embedding_dim)
        self.cat_layers = []
        for idx in np.arange(len(self.cat_layer_dims)-1):
            in_features = self.cat_layer_dims[idx]
            out_features = self.cat_layer_dims[idx+1]
            self.cat_layers.append(nn.Linear(in_features, out_features))
            self.cat_layers.append(nn.ReLU())
        self.cat_forward = nn.Sequential(*self.cat_layers)

        self.deep_layer_dims = config['deep_layers']
        self.deep_layer_dims.insert(0, self.player_embedding_dim*3)
        self.deep_layers = []
        for idx in np.arange(len(self.deep_layer_dims)-1):
            in_features = self.deep_layer_dims[idx]
            out_features = self.deep_layer_dims[idx+1]
            self.deep_layers.append(nn.Linear(in_features, out_features))
            self.deep_layers.append(nn.ReLU())

        self.deep_forward = nn.Sequential(*self.deep_layers)

        self.affine_output = nn.Linear(self.deep_layer_dims[-1], 1)

    def forward(self, variables):

        w = variables[:, 0].long()
        period = variables[:, 1].long()
        pts = variables[:, 2].long()
        defender = variables[:, 3].long()
        player = variables[:, 4].long()

        cont = variables[:, 5:]

        w_emb = self.w_embedding(w)
        per_emb = self.period_embedding(period)
        pts_emb = self.pts_type_embedding(pts)
        def_emb = self.defender_embedding(defender)
        player_emb = self.player_embedding(player)

        cat_emb = torch.cat((w_emb, per_emb, pts_emb, def_emb), 1)
        cat_out = self.cat_forward(cat_emb)
        cont_out = self.cont_forward(cont)

        out = torch.cat((cat_out, cont_out), 1)

        player_effect = out*player_emb
        total_out = torch.cat((player_effect, out, player_emb), 1)

        out = self.deep_forward(total_out)
        affine = self.affine_output(out)
        return affine

class Engine:
    def __init__(self, model, config, utils):
        self.utils = utils
        self.device = config['device']
        if self.device == 'cuda':
            self.model = model.cuda()
        else:
            self.model = model
        self.num_epochs = config['num_epochs']
        self.learning_rate = config['learning_rate']
        self.optim_name = config['optim']
        self.weight_decay = config['weight_decay']
        self.momentum = config['momentum']
        self.criterion = config['criterion']
        self.sigmoid = nn.Sigmoid()

    def create_optimizer(self):
        if self.optim_name == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        elif self.optim_name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise TypeError('Optimizer not supported yet. Please pick SGD or Adam')

    def train_one_batch(self, x, y):
        if self.device == 'cuda':
            x = x.cuda()
            y = y.cuda()
        self.optimizer.zero_grad()
        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()
        preds = self.sigmoid(outputs.detach()) > 0.5
        if self.device == 'cuda':
            return loss.item(), list(preds.cpu().numpy()), list(y.cpu().detach().numpy())
        else:
            return loss.item(), list(preds.numpy()), list(y.detach().numpy())

    def train_one_epoch(self, epoch_num):

        epoch_loss = 0
        all_preds = []
        all_labels = []
        for x,y in self.utils.train_loader:
            if self.device == 'cuda':
                x = x.cuda()
                y = y.cuda()
            l, preds, labels = self.train_one_batch(x, y)
            epoch_loss += l
            all_preds += preds
            all_labels += labels

        print('Train Scores:')
        print('Epoch:', epoch_num, 'Loss:', epoch_loss, 'Accuracy:', accuracy_score(all_labels, all_preds))

    def test_one_epoch(self, epoch_num):

        epoch_loss = 0
        all_preds = []
        all_labels = []
        for x, y in self.utils.test_loader:
            outputs = self.model(x)
            preds = self.sigmoid(outputs.detach()) > 0.5
            if self.device == 'cpu':
                preds = list(preds.cpu().numpy())
            else:
                preds = list(preds.numpy())
            l = self.criterion(outputs, y).item()
            if self.device == 'cuda':
                labels = list(y.cpu().detach().numpy())
            else:
                labels = list(y.detach().numpy())
            epoch_loss += l
            all_preds += preds
            all_labels += labels

        print('Test Scores:')
        print('Epoch:', epoch_num, 'Loss:', epoch_loss, 'Accuracy:', accuracy_score(all_labels, all_preds))
        return epoch_loss

    def train_to_completion(self):
        self.create_optimizer()
        best_loss = 1e10
        self.best_model = None
        tolerance = 10
        breaking = 0
        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch)
            l = self.test_one_epoch(epoch)
            if l < best_loss:
                best_loss = l
                self.best_model = deepcopy(self.model.state_dict())
                breaking = 0
            else:
                breaking += 1
                if breaking >= tolerance:
                    break

        print('Training stopped at epoch {}'.format(epoch))
        print('Best testing loss = {}'.format(best_loss))


