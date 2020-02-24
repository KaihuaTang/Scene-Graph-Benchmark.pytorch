# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.utils import cat
from .utils_motifs import obj_edge_vectors, center_x, sort_by_score, to_onehot, get_dropout_mask, nms_overlaps, encode_box_info, generate_attributes_target, normalize_sigmoid_logits


class AttributeDecoderRNN(nn.Module):
    def __init__(self, config, obj_classes, att_classes, embed_dim, inputs_dim, hidden_dim, rnn_drop):
        super(AttributeDecoderRNN, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.att_classes = att_classes
        self.embed_dim = embed_dim
        self.max_num_attri = config.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES
        self.num_attri_cat = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES

        obj_embed_vecs = obj_edge_vectors(['start'] + self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=embed_dim)
        att_embed_vecs = obj_edge_vectors(self.att_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=embed_dim)
        self.obj_embed = nn.Embedding(len(self.obj_classes)+1, embed_dim)
        self.att_embed = nn.Embedding(len(self.att_classes), embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.att_embed.weight.copy_(att_embed_vecs, non_blocking=True)

        self.hidden_size = hidden_dim
        self.inputs_dim = inputs_dim
        self.input_size = self.inputs_dim + self.embed_dim * 2
        self.nms_thresh = 0.3
        self.rnn_drop=rnn_drop

        self.input_linearity = torch.nn.Linear(self.input_size, 6 * self.hidden_size, bias=True)
        self.state_linearity = torch.nn.Linear(self.hidden_size, 5 * self.hidden_size, bias=True)
        self.out_obj = nn.Linear(self.hidden_size, len(self.obj_classes))
        self.out_att = nn.Linear(self.hidden_size, len(self.att_classes))
        
        self.init_parameters()

    def init_parameters(self):
        # Use sensible default initializations for parameters.
        with torch.no_grad():
            torch.nn.init.constant_(self.state_linearity.bias, 0.0)
            torch.nn.init.constant_(self.input_linearity.bias, 0.0)

    def lstm_equations(self, timestep_input, previous_state, previous_memory, dropout_mask=None):
        """
        Does the hairy LSTM math
        :param timestep_input:
        :param previous_state:
        :param previous_memory:
        :param dropout_mask:
        :return:
        """
        # Do the projections for all the gates all at once.
        projected_input = self.input_linearity(timestep_input)
        projected_state = self.state_linearity(previous_state)

        # Main LSTM equations using relevant chunks of the big linear
        # projections of the hidden state and inputs.
        input_gate = torch.sigmoid(projected_input[:, 0 * self.hidden_size:1 * self.hidden_size] +
                                   projected_state[:, 0 * self.hidden_size:1 * self.hidden_size])
        forget_gate = torch.sigmoid(projected_input[:, 1 * self.hidden_size:2 * self.hidden_size] +
                                    projected_state[:, 1 * self.hidden_size:2 * self.hidden_size])
        memory_init = torch.tanh(projected_input[:, 2 * self.hidden_size:3 * self.hidden_size] +
                                 projected_state[:, 2 * self.hidden_size:3 * self.hidden_size])
        output_gate = torch.sigmoid(projected_input[:, 3 * self.hidden_size:4 * self.hidden_size] +
                                    projected_state[:, 3 * self.hidden_size:4 * self.hidden_size])
        memory = input_gate * memory_init + forget_gate * previous_memory
        timestep_output = output_gate * torch.tanh(memory)

        highway_gate = torch.sigmoid(projected_input[:, 4 * self.hidden_size:5 * self.hidden_size] +
                                         projected_state[:, 4 * self.hidden_size:5 * self.hidden_size])
        highway_input_projection = projected_input[:, 5 * self.hidden_size:6 * self.hidden_size]
        timestep_output = highway_gate * timestep_output + (1 - highway_gate) * highway_input_projection

        # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
        if dropout_mask is not None and self.training:
            timestep_output = timestep_output * dropout_mask
        return timestep_output, memory

    def forward(self, inputs, initial_state=None, labels=None, boxes_for_nms=None):
        if not isinstance(inputs, PackedSequence):
            raise ValueError('inputs must be PackedSequence but got %s' % (type(inputs)))

        assert isinstance(inputs, PackedSequence)
        sequence_tensor, batch_lengths, _, _ = inputs
        batch_size = batch_lengths[0]

        # We're just doing an LSTM decoder here so ignore states, etc
        if initial_state is None:
            previous_memory = sequence_tensor.new().resize_(batch_size, self.hidden_size).fill_(0)
            previous_state = sequence_tensor.new().resize_(batch_size, self.hidden_size).fill_(0)
        else:
            assert len(initial_state) == 2
            previous_memory = initial_state[1].squeeze(0)
            previous_state = initial_state[0].squeeze(0)

        previous_obj_embed = self.obj_embed.weight[0, None].expand(batch_size, self.embed_dim)
        previous_att_embed = self.att_embed.weight[0, None].expand(batch_size, self.embed_dim) # use background as start

        if self.rnn_drop > 0.0:
            dropout_mask = get_dropout_mask(self.rnn_drop, previous_memory.size(), previous_memory.device)
        else:
            dropout_mask = None

        # Only accumulating label predictions here, discarding everything else
        out_dists = []
        att_dists = []
        out_commitments = []

        end_ind = 0
        for i, l_batch in enumerate(batch_lengths):
            start_ind = end_ind
            end_ind = end_ind + l_batch

            if previous_memory.size(0) != l_batch:
                previous_memory = previous_memory[:l_batch]
                previous_state = previous_state[:l_batch]
                previous_obj_embed = previous_obj_embed[:l_batch]
                previous_att_embed = previous_att_embed[:l_batch]
                if dropout_mask is not None:
                    dropout_mask = dropout_mask[:l_batch]

            timestep_input = torch.cat((sequence_tensor[start_ind:end_ind], previous_obj_embed, previous_att_embed), 1)

            previous_state, previous_memory = self.lstm_equations(timestep_input, previous_state,
                                                                  previous_memory, dropout_mask=dropout_mask)

            pred_dist = self.out_obj(previous_state)
            attr_dist = self.out_att(previous_state)
            out_dists.append(pred_dist)
            att_dists.append(attr_dist)

            if self.training:
                labels_to_embed = labels[start_ind:end_ind].clone()
                # Whenever labels are 0 set input to be our max prediction
                nonzero_pred = pred_dist[:, 1:].max(1)[1] + 1
                is_bg = (labels_to_embed == 0).nonzero()
                if is_bg.dim() > 0:
                    labels_to_embed[is_bg.squeeze(1)] = nonzero_pred[is_bg.squeeze(1)]
                out_commitments.append(labels_to_embed)
                previous_obj_embed = self.obj_embed(labels_to_embed+1)
            else:
                assert l_batch == 1
                out_dist_sample = F.softmax(pred_dist, dim=1)
                best_ind = out_dist_sample[:, 1:].max(1)[1] + 1
                out_commitments.append(best_ind)
                previous_obj_embed = self.obj_embed(best_ind+1)

        previous_att_embed = normalize_sigmoid_logits(attr_dist) @ self.att_embed.weight

        # Do NMS here as a post-processing step
        if boxes_for_nms is not None and not self.training:
            is_overlap = nms_overlaps(boxes_for_nms).view(
                boxes_for_nms.size(0), boxes_for_nms.size(0), boxes_for_nms.size(1)
            ).cpu().numpy() >= self.nms_thresh

            out_dists_sampled = F.softmax(torch.cat(out_dists,0), 1).cpu().numpy()
            out_dists_sampled[:,0] = 0

            out_commitments = out_commitments[0].new(len(out_commitments)).fill_(0)

            for i in range(out_commitments.size(0)):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_commitments[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample

            out_commitments = out_commitments
        else:
            out_commitments = torch.cat(out_commitments, 0)

        return torch.cat(out_dists, 0), out_commitments, torch.cat(att_dists, 0)


class AttributeLSTMContext(nn.Module):
    """
    Modified from neural-motifs to encode contexts for each objects
    """
    def __init__(self, config, obj_classes, att_classes, rel_classes, in_channels):
        super(AttributeLSTMContext, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.att_classes = att_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)
        self.num_att_classes = len(att_classes)
        self.max_num_attri = config.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES
        self.num_attri_cat = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # word embedding
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        att_embed_vecs = obj_edge_vectors(self.att_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.att_embed1 = nn.Embedding(self.num_att_classes, self.embed_dim)
        self.att_embed2 = nn.Embedding(self.num_att_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.att_embed1.weight.copy_(att_embed_vecs, non_blocking=True)
            self.att_embed2.weight.copy_(att_embed_vecs, non_blocking=True)

        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])

        # object & relation context
        self.obj_dim = in_channels
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nl_obj = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER
        self.nl_edge = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_REL_LAYER
        assert self.nl_obj > 0 and self.nl_edge > 0

        # TODO Kaihua Tang
        # AlternatingHighwayLSTM is invalid for pytorch 1.0
        self.obj_ctx_rnn = torch.nn.LSTM(
                input_size=self.obj_dim+self.embed_dim*2 + 128,
                hidden_size=self.hidden_dim,
                num_layers=self.nl_obj,
                dropout=self.dropout_rate if self.nl_obj > 1 else 0,
                bidirectional=True)
        self.decoder_rnn = AttributeDecoderRNN(self.cfg, self.obj_classes, self.att_classes, embed_dim=self.embed_dim,
                inputs_dim=self.hidden_dim + self.obj_dim + self.embed_dim*2 + 128,
                hidden_dim=self.hidden_dim,
                rnn_drop=self.dropout_rate)
        self.edge_ctx_rnn = torch.nn.LSTM(
                input_size=self.embed_dim*2 + self.hidden_dim + self.obj_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.nl_edge,
                dropout=self.dropout_rate if self.nl_edge > 1 else 0,
                bidirectional=True)
        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.lin_obj_h = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.lin_edge_h = nn.Linear(self.hidden_dim*2, self.hidden_dim)

    def sort_rois(self, proposals):
        c_x = center_x(proposals)
        # leftright order
        scores = c_x / (c_x.max() + 1)
        return sort_by_score(proposals, scores)

    def obj_ctx(self, obj_feats, proposals, obj_labels=None, att_labels=None, boxes_per_cls=None):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_labels: [num_obj] the GT labels of the image
        :param box_priors: [num_obj, 4] boxes. We'll use this for NMS
        :param boxes_per_cls
        :return: obj_dists: [num_obj, #classes] new probability distribution.
                 obj_preds: argmax of that distribution.
                 obj_final_ctx: [num_obj, #feats] For later!
        """
        # Sort by the confidence of the maximum detection.
        perm, inv_perm, ls_transposed = self.sort_rois(proposals)
        # Pass object features, sorted by score, into the encoder LSTM
        obj_inp_rep = obj_feats[perm].contiguous()
        input_packed = PackedSequence(obj_inp_rep, ls_transposed)
        encoder_rep = self.obj_ctx_rnn(input_packed)[0][0]
        encoder_rep = self.lin_obj_h(encoder_rep) # map to hidden_dim
        # Decode in order
        if self.mode != 'predcls':
            decoder_inp = PackedSequence(torch.cat((obj_inp_rep, encoder_rep), 1),
                                         ls_transposed)
            obj_dists, obj_preds, att_dists = self.decoder_rnn(
                decoder_inp, #obj_dists[perm],
                labels=obj_labels[perm] if obj_labels is not None else None,
                boxes_for_nms=boxes_per_cls[perm] if boxes_per_cls is not None else None,
                )
            obj_preds = obj_preds[inv_perm]
            obj_dists = obj_dists[inv_perm]
            att_dists = att_dists[inv_perm]
        else:
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
            att_dists, att_fg_ind = generate_attributes_target(att_labels, att_labels.device, self.max_num_attri, self.num_attri_cat)
        encoder_rep = encoder_rep[inv_perm]

        return obj_dists, obj_preds, att_dists, encoder_rep, perm, inv_perm, ls_transposed

    def edge_ctx(self, obj_feats, obj_preds, att_dists, perm, inv_perm, ls_transposed):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :return: edge_ctx: [num_obj, #feats] For later!
        """
        obj_embed2 = self.obj_embed2(obj_preds)        
        att_embed2 = normalize_sigmoid_logits(att_dists) @ self.att_embed2.weight
        inp_feats = torch.cat((obj_embed2, att_embed2, obj_feats), 1)

        edge_input_packed = PackedSequence(inp_feats[perm], ls_transposed)
        edge_reps = self.edge_ctx_rnn(edge_input_packed)[0][0]
        edge_reps = self.lin_edge_h(edge_reps) # map to hidden_dim

        edge_ctx = edge_reps[inv_perm]
        return edge_ctx

    def forward(self, x, proposals, logger=None):
        # labels will be used in DecoderRNN during training (for nms)
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            att_labels = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)
        else:
            obj_labels = None
            att_labels = None

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_labels)
            gt_att_labels = self.generate_attributes_target(att_labels)
            gt_att_labels = gt_att_labels / (gt_att_labels.sum(1).unsqueeze(-1) + 1e-12)
            att_embed = gt_att_labels @ self.att_embed1.weight
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            att_logits = cat([proposal.get_field("attribute_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight
            att_embed = normalize_sigmoid_logits(att_logits) @ self.att_embed1.weight
        
        assert proposals[0].mode == 'xyxy'
        pos_embed = self.pos_embed(encode_box_info(proposals))
        obj_pre_rep = cat((x, obj_embed, att_embed, pos_embed), -1)

        boxes_per_cls = None
        if self.mode == 'sgdet' and not self.training:
            boxes_per_cls = cat([proposal.get_field('boxes_per_cls') for proposal in proposals], dim=0) # comes from post process of box_head

        # object level contextual feature
        obj_dists, obj_preds, att_dists, obj_ctx, perm, inv_perm, ls_transposed = self.obj_ctx(obj_pre_rep, proposals, obj_labels, att_labels, boxes_per_cls)
        # edge level contextual feature
        obj_rel_rep = cat((x, obj_ctx), -1)
        edge_ctx = self.edge_ctx(obj_rel_rep, obj_preds=obj_preds, att_dists=att_dists, perm=perm, 
                                inv_perm=inv_perm, ls_transposed=ls_transposed)

        return obj_dists, obj_preds, att_dists, edge_ctx

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        max_num_attri = attributes.shape[1]
        num_obj = attributes.shape[0]

        with_attri_idx = (attributes.sum(-1) > 0).long()
        without_attri_idx = 1 - with_attri_idx
        
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=attributes.device).float()
      
        for idx in torch.nonzero(with_attri_idx).squeeze(1).tolist():
            for k in range(max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1

        return attribute_targets
