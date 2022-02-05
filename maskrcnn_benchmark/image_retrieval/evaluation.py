from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import time
import datetime
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.utils.data as data
from torch.nn.utils import weight_norm
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
import pandas as pd
import itertools


def evaluator(logger, input_lists):
    cat_data = []
    for item in input_lists:
        cat_data.append(item[0])
    # shape [num_image, 2, hidden_dim]
    cat_data = torch.cat(cat_data, dim=0).squeeze(2)

    img_graph_embeds = cat_data[:, 0, :]
    txt_graph_embeds = cat_data[:, 1, :]

    similarity = img_graph_embeds @ txt_graph_embeds.T

    norm_1 = img_graph_embeds.norm(dim=1, p=2)
    norm_2 = txt_graph_embeds.norm(dim=1, p=2)
    norm = norm_1.unsqueeze(1) * norm_2.unsqueeze(1).T
    #Normalize the similarity scores to make them comparable
    similarity = similarity / norm

    pred_rank = (similarity >= similarity.diag().view(-1, 1)).sum(-1)
    #pred_rank = (similarity >= (torch.ones_like(similarity.diag()) * similarity.diag().mean()).view(-1, 1)).sum(-1)
    num_sample = pred_rank.shape[0]
    thres = [1, 5, 10, 20, 50, 100]
    for k in thres:
        logger.info('Recall @ %d: %.4f; ' % (k, float((pred_rank<k).sum()) / num_sample))

    return similarity


def calculate_normalized_cosine_similarity_for_captions(input):
    """
    Input has the dimensions: number of entries * 2 * vector dimension
    :param input:
    :return:
    """

    first_embeds = input[:, 0, :]
    second_embeds = input[:, 1, :]

    return calculate_normalized_cosine_similarity(first_embeds, second_embeds)


def calculate_normalized_cosine_similarity(gallery_input, query):
    """
    Input has the dimensions: number of entries X * vector dimension
    query has the dimension number of entries Y * vector dimension
    :param input:
    :return:
    """

    # Trivial check to insure the dimension stated in the method header.
    num_dim_gallery = len(gallery_input.shape)
    num_dim_query = len(query.shape)
    assert num_dim_gallery < 3
    assert num_dim_query < 3
    if num_dim_query == 1:
        query = query.unsqueeze(0)
    if num_dim_gallery == 1:
        gallery_input = gallery_input.unsqueeze(0)
    assert gallery_input.shape[1] == query.shape[1]

    similarity = gallery_input @ query.T

    norm_1 = gallery_input.norm(dim=1, p=2)
    norm_2 = query.norm(dim=1, p=2)
    norm = norm_1.unsqueeze(1) * norm_2.unsqueeze(1).T
    # Normalize the similarity scores to make them comparable
    similarity = similarity / norm

    return similarity


def compute_recall_johnson_feiefei(similarity, threshold,  recall_at: list = [1, 2, 3, 4, 5, 10, 20, 50, 100]):
    """
        This is how I understood recall computation from  https://openaccess.thecvf.com/content_cvpr_2015/papers/Johnson_Image_Retrieval_Using_2015_CVPR_paper.pdf, p.6
        For each image, we know what is the expected best result (gold image) for a given text query.
        If we consider the best k results as in @K metrics, we have to check if our image is included in the top k.
        We can then compute a proportion of times our system includes the true image and also calculate the mean rank recomandation
        of the gold image.

    :param similarity:
    :param threshold:
    :param category: Not used. Just to keep the same signature with compute_recall_on_category
    :param recall_at:
    :return:
    """
    number_entries = similarity.shape[0]
    values, ranks = torch.topk(similarity, number_entries)
    gold_recommendations = torch.arange(0, number_entries, dtype=ranks.dtype, device=ranks.device).unsqueeze(1)

    # dimension 0 is the entry dimension, dimension 1 is the ranking for a given entry
    entry_ranks, gold_ranks = (ranks == gold_recommendations).nonzero(as_tuple=True)
    mean_rank = (gold_ranks + 1).type(torch.float).mean()

    if threshold:
        threshold_mask = (values >= threshold)
        # Due to the threshold, you might have less entries returned than number_entries
        entry_ranks, gold_ranks = torch.logical_and((ranks == gold_recommendations), threshold_mask).nonzero(
            as_tuple=True)

    recall_val = {"recall_at_" + str(k): ((gold_ranks < k).sum().type(torch.float) / number_entries).to("cpu").numpy()
                  for k in recall_at if
                  k <= number_entries}

    mean_rank = mean_rank.to("cpu").numpy()

    return recall_val, mean_rank

def compute_similarity(stacked_vectors, threshold=None):
    """
    Compute the average of all pairwise combination of captions
    :param ade20k_split:
    :param threshold:
    :param recall_funct:
    :return:
    """


    num_captions = stacked_vectors.shape[1]
    k = 2

    all_caption_pairs = list(itertools.combinations(range(num_captions), k))
    recall_list = []
    mean_rank_list = []
    similarity_list = []
    for pair in all_caption_pairs:
        similarity = calculate_normalized_cosine_similarity_for_captions(stacked_vectors[:, pair,:])
        recall_val, mean_rank = compute_recall_johnson_feiefei(similarity, threshold)
        similarity_list.append(similarity.diag().mean().to("cpu").numpy())
        recall_list.append(recall_val)
        mean_rank_list.append(mean_rank)


    recall_mean = pd.DataFrame(recall_list).mean().to_dict()
    average_mean_rank = pd.DataFrame(mean_rank_list).mean()[0]
    average_similarity = pd.DataFrame(similarity_list).mean()[0]

    for k in recall_mean.keys():
        print(f"{k}: {recall_mean[k]}")

    recall_mean["mean_rank"] = average_mean_rank
    print(f"Mean Rank: {average_mean_rank}")


    print(f"Average Similarity: {average_similarity}")

    recall_mean["average_similarity"] = average_similarity
    recall_mean["threshold"] = threshold
    return recall_mean

def run_evaluation(evaluation_name, split,  threshold_list,  output_dir):
    values = []
    print(f"\n############## Start Evaluation: {evaluation_name} ############## ")
    for t in threshold_list:
        print("\n")
        print(f"Threshold: {t}")
        val = compute_similarity(split, t)
        values.append(val)
        print("\n")
    print(f"############## End Evaluation: {evaluation_name} ############## ")
    df = pd.DataFrame(values)
    output_path = os.path.join(output_dir, evaluation_name + ".csv")
    print(f"Saving data to {output_path}")
    df.to_csv(output_path)
