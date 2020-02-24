import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .utils_motifs import get_dropout_mask
from .utils_relation import block_orthogonal

class MultiLayer_BTreeLSTM(nn.Module):
    """
    Multilayer Bidirectional Tree LSTM
    Each layer contains one forward lstm(leaves to root) and one backward lstm(root to leaves)
    """
    def __init__(self, in_dim, out_dim, num_layer, dropout=0.0):
        super(MultiLayer_BTreeLSTM, self).__init__()
        self.num_layer = num_layer
        layers = []
        layers.append(BidirectionalTreeLSTM(in_dim, out_dim, dropout))
        for i in range(num_layer - 1):
            layers.append(BidirectionalTreeLSTM(out_dim, out_dim, dropout))
        self.multi_layer_lstm = nn.ModuleList(layers)

    def forward(self, tree, features, num_obj):
        for i in range(self.num_layer):
            features = self.multi_layer_lstm[i](tree, features, num_obj)
        return features


class BidirectionalTreeLSTM(nn.Module):
    """
    Bidirectional Tree LSTM
    Contains one forward lstm(leaves to root) and one backward lstm(root to leaves)
    Dropout mask will be generated one time for all trees in the forest, to make sure the consistancy
    """
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super(BidirectionalTreeLSTM, self).__init__()
        self.dropout = dropout
        self.out_dim = out_dim
        self.treeLSTM_foreward = OneDirectionalTreeLSTM(in_dim, int(out_dim / 2), 'foreward', dropout)
        self.treeLSTM_backward = OneDirectionalTreeLSTM(in_dim, int(out_dim / 2), 'backward', dropout)

    def forward(self, tree, features, num_obj):
        foreward_output = self.treeLSTM_foreward(tree, features, num_obj)
        backward_output = self.treeLSTM_backward(tree, features, num_obj)
    
        final_output = torch.cat((foreward_output, backward_output), 1)

        return final_output


class OneDirectionalTreeLSTM(nn.Module):
    """
    One Way Tree LSTM
    direction = foreward | backward
    """
    def __init__(self, in_dim, out_dim, direction, dropout=0.0):
        super(OneDirectionalTreeLSTM, self).__init__()
        self.dropout = dropout
        self.out_dim = out_dim
        if direction == 'foreward':
            self.treeLSTM = BiTreeLSTM_Foreward(in_dim, out_dim)
        elif direction == 'backward':
            self.treeLSTM = BiTreeLSTM_Backward(in_dim, out_dim)
        else:
            print('Error Tree LSTM Direction')

    def forward(self, tree, features, num_obj):
        # calc dropout mask, same for all
        if self.dropout > 0.0:
            dropout_mask = get_dropout_mask(self.dropout, (1, self.out_dim), features.device)
        else:
            dropout_mask = None

        # tree lstm input
        h_order = torch.tensor([0] * num_obj, device=features.device, dtype=torch.int64) # used to resume order
        lstm_io = TreeLSTM_IO(None, h_order, 0, None, None, dropout_mask)
        # run tree lstm forward (leaves to root)
        self.treeLSTM(tree, features, lstm_io)
        # resume order to the same as input
        output = lstm_io.hidden[lstm_io.order.long()]
        return output


class BiTreeLSTM_Foreward(nn.Module):
    """
    From leaves to root
    """
    def __init__(self, feat_dim, h_dim, is_pass_embed=False, embed_layer=None, embed_out_layer=None):
        super(BiTreeLSTM_Foreward, self).__init__()
        self.feat_dim = feat_dim
        self.h_dim = h_dim
        self.is_pass_embed = is_pass_embed
        self.embed_layer = embed_layer
        self.embed_out_layer = embed_out_layer

        self.px = nn.Linear(self.feat_dim, self.h_dim)
        self.ioffux = nn.Linear(self.feat_dim, 6 * self.h_dim)
        self.ioffuh_left = nn.Linear(self.h_dim, 6 * self.h_dim)
        self.ioffuh_right = nn.Linear(self.h_dim, 6 * self.h_dim)

        # initialization
        with torch.no_grad():
            block_orthogonal(self.px.weight, [self.h_dim, self.feat_dim])
            block_orthogonal(self.ioffux.weight, [self.h_dim, self.feat_dim])
            block_orthogonal(self.ioffuh_left.weight, [self.h_dim, self.h_dim])
            block_orthogonal(self.ioffuh_right.weight, [self.h_dim, self.h_dim])

            self.px.bias.fill_(0.0)
            self.ioffux.bias.fill_(0.0)
            self.ioffuh_left.bias.fill_(0.0)
            self.ioffuh_right.bias.fill_(0.0)
            # Initialize forget gate biases to 1.0 as per An Empirical
            # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
            self.ioffuh_left.bias[2 * self.h_dim:4 * self.h_dim].fill_(0.5)
            self.ioffuh_right.bias[2 * self.h_dim:4 * self.h_dim].fill_(0.5)


    def node_forward(self, feat_inp, left_c, right_c, left_h, right_h, dropout_mask):
        projected_x = self.px(feat_inp)
        ioffu = self.ioffux(feat_inp) + self.ioffuh_left(left_h) + self.ioffuh_right(right_h)
        i, o, f_l, f_r, u, r = torch.split(ioffu, ioffu.size(1) // 6, dim=1)
        i, o, f_l, f_r, u, r = torch.sigmoid(i), torch.sigmoid(o), torch.sigmoid(f_l), torch.sigmoid(f_r), torch.tanh(u), torch.sigmoid(r)

        c = torch.mul(i, u) + torch.mul(f_l, left_c) + torch.mul(f_r, right_c)
        h = torch.mul(o, torch.tanh(c))
        h_final = torch.mul(r, h) + torch.mul((1 - r), projected_x)
        # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
        if dropout_mask is not None and self.training:
            h_final = torch.mul(h_final, dropout_mask)
        return c, h_final

    def forward(self, tree, features, treelstm_io):
        """
        tree: The root for a tree
        features: [num_obj, featuresize]
        treelstm_io.hidden: init as None, cat until it covers all objects as [num_obj, hidden_size]
        treelstm_io.order: init as 0 for all [num_obj], update for recovering original order
        """
        # recursively search child
        if tree.left_child is not None:
            self.forward(tree.left_child, features, treelstm_io)
        if tree.right_child is not None:
            self.forward(tree.right_child, features, treelstm_io)
        # get c,h from left child
        if tree.left_child is None:
            left_c = torch.tensor([0.0] * self.h_dim, device=features.device).float().view(1,-1)
            left_h = torch.tensor([0.0] * self.h_dim, device=features.device).float().view(1,-1)
            # Only being used in decoder network
            if self.is_pass_embed:
                left_embed = self.embed_layer.weight[0]
        else:
            left_c = tree.left_child.state_c
            left_h = tree.left_child.state_h
            # Only being used in decoder network
            if self.is_pass_embed:
                left_embed = tree.left_child.embeded_label
        # get c,h from right child
        if tree.right_child is None:
            right_c = torch.tensor([0.0] * self.h_dim, device=features.device).float().view(1,-1)
            right_h = torch.tensor([0.0] * self.h_dim, device=features.device).float().view(1,-1)
            # Only being used in decoder network
            if self.is_pass_embed: 
                right_embed = self.embed_layer.weight[0]
        else:
            right_c = tree.right_child.state_c
            right_h = tree.right_child.state_h
            # Only being used in decoder network
            if self.is_pass_embed: 
                right_embed = tree.right_child.embeded_label
        # Only being used in decoder network
        if self.is_pass_embed: 
            next_feature = torch.cat((features[tree.index].view(1, -1), left_embed.view(1,-1), right_embed.view(1,-1)), 1)
        else:
            next_feature = features[tree.index].view(1, -1)

        c, h = self.node_forward(next_feature, left_c, right_c, left_h, right_h, treelstm_io.dropout_mask)
        tree.state_c = c
        tree.state_h = h
        # record label prediction
        # Only being used in decoder network
        if self.is_pass_embed: 
            pass_embed_postprocess(h, self.embed_out_layer, self.embed_layer, tree, treelstm_io, self.training)

        # record hidden state
        if treelstm_io.hidden is None:
            treelstm_io.hidden = h.view(1, -1)
        else:
            treelstm_io.hidden = torch.cat((treelstm_io.hidden, h.view(1, -1)), 0)

        treelstm_io.order[tree.index] = treelstm_io.order_count
        treelstm_io.order_count += 1
        return


class BiTreeLSTM_Backward(nn.Module):
    """
    from root to leaves
    """
    def __init__(self, feat_dim, h_dim, is_pass_embed=False, embed_layer=None, embed_out_layer=None):
        super(BiTreeLSTM_Backward, self).__init__()
        self.feat_dim = feat_dim
        self.h_dim = h_dim
        self.is_pass_embed = is_pass_embed
        self.embed_layer = embed_layer
        self.embed_out_layer = embed_out_layer

        self.px = nn.Linear(self.feat_dim, self.h_dim)
        self.iofux = nn.Linear(self.feat_dim, 5 * self.h_dim)
        self.iofuh = nn.Linear(self.h_dim, 5 * self.h_dim)
        
        # initialization
        with torch.no_grad():
            block_orthogonal(self.px.weight, [self.h_dim, self.feat_dim])
            block_orthogonal(self.iofux.weight, [self.h_dim, self.feat_dim])
            block_orthogonal(self.iofuh.weight, [self.h_dim, self.h_dim])

            self.px.bias.fill_(0.0)
            self.iofux.bias.fill_(0.0)
            self.iofuh.bias.fill_(0.0)
            # Initialize forget gate biases to 1.0 as per An Empirical
            # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
            self.iofuh.bias[2 * self.h_dim:3 * self.h_dim].fill_(1.0)

    def node_backward(self, feat_inp, root_c, root_h, dropout_mask):
        
        projected_x = self.px(feat_inp)
        iofu = self.iofux(feat_inp) + self.iofuh(root_h)
        i, o, f, u, r = torch.split(iofu, iofu.size(1) // 5, dim=1)
        i, o, f, u, r = torch.sigmoid(i), torch.sigmoid(o), torch.sigmoid(f), torch.tanh(u), torch.sigmoid(r)

        c = torch.mul(i, u) + torch.mul(f, root_c)
        h = torch.mul(o, torch.tanh(c))
        h_final = torch.mul(r, h) + torch.mul((1 - r), projected_x)
        # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
        if dropout_mask is not None and self.training:
            h_final = torch.mul(h_final, dropout_mask)
        return c, h_final

    def forward(self, tree, features, treelstm_io):
        """
        tree: The root for a tree
        features: [num_obj, featuresize]
        treelstm_io.hidden: init as None, cat until it covers all objects as [num_obj, hidden_size]
        treelstm_io.order: init as 0 for all [num_obj], update for recovering original order
        """

        if tree.parent is None:
            root_c = torch.tensor([0.0] * self.h_dim, device=features.device).float().view(1,-1)
            root_h = torch.tensor([0.0] * self.h_dim, device=features.device).float().view(1,-1)
            if self.is_pass_embed:
                root_embed = self.embed_layer.weight[0]
        else:
            root_c = tree.parent.state_c_backward
            root_h = tree.parent.state_h_backward
            if self.is_pass_embed:
                root_embed = tree.parent.embeded_label

        if self.is_pass_embed:
            next_features = torch.cat((features[tree.index].view(1, -1), root_embed.view(1,-1)), 1)
        else:
            next_features = features[tree.index].view(1, -1)

        c, h = self.node_backward(next_features, root_c, root_h, treelstm_io.dropout_mask)
        tree.state_c_backward = c
        tree.state_h_backward = h
        # record label prediction
        # Only being used in decoder network
        if self.is_pass_embed: 
            pass_embed_postprocess(h, self.embed_out_layer, self.embed_layer, tree, treelstm_io, self.training)

        # record hidden state
        if treelstm_io.hidden is None:
            treelstm_io.hidden = h.view(1, -1)
        else:
            treelstm_io.hidden = torch.cat((treelstm_io.hidden, h.view(1, -1)), 0)
 
        treelstm_io.order[tree.index] = treelstm_io.order_count
        treelstm_io.order_count += 1

        # recursively update from root to leaves
        if tree.left_child is not None:
            self.forward(tree.left_child, features, treelstm_io)
        if tree.right_child is not None:
            self.forward(tree.right_child, features, treelstm_io)

        return


def pass_embed_postprocess(h, embed_out_layer, embed_layer, tree, treelstm_io, is_training):
    """
    Calculate districution and predict/sample labels
    Add to lstm_IO
    """
    pred_dist = embed_out_layer(h)
    label_to_embed = F.softmax(pred_dist.view(-1), 0)[1:].max(0)[1] + 1
    if is_training:
        sampled_label = F.softmax(pred_dist.view(-1), 0)[1:].multinomial(1).detach() + 1
        tree.embeded_label = embed_layer(sampled_label+1)
    else:
        tree.embeded_label = embed_layer(label_to_embed+1)

    if treelstm_io.dists is None:
        treelstm_io.dists = pred_dist.view(1, -1)
    else:
        treelstm_io.dists = torch.cat((treelstm_io.dists, pred_dist.view(1, -1)), 0)

    if treelstm_io.commitments is None:
        treelstm_io.commitments = label_to_embed.view(-1)
    else:
        treelstm_io.commitments = torch.cat((treelstm_io.commitments, label_to_embed.view(-1)), 0)


class TreeLSTM_IO(object):
    def __init__(self, hidden_tensor, order_tensor, order_count, dists_tensor, commitments_tensor, dropout_mask):
        self.hidden = hidden_tensor # Float tensor [num_obj, self.out_dim]
        self.order = order_tensor # Long tensor [num_obj]
        self.order_count = order_count # int
        self.dists = dists_tensor # FLoat tensor [num_obj, len(self.classes)]
        self.commitments = commitments_tensor
        self.dropout_mask = dropout_mask