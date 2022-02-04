import argparse
import os
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
import sng_parser
import json
from tqdm import tqdm
import collections
import torch
import torchtext as tt
import spacy

def make_vocab(all_caps, outpath, file_name_entity, file_name_relation, freq=1):
    counter_entity = collections.Counter()
    counter_relation = collections.Counter()
    # result = tt.vocab.Vocab(counter_obj, min_freq=1)

    ent = os.path.join(outpath, file_name_entity)
    rel = os.path.join(outpath, file_name_relation)
    if not os.path.exists(ent) or not os.path.exists(rel):
        print("Generating Vocabulary.")
        for k in tqdm(all_caps.keys()):
            caps = all_caps[k]
            raw_graphs = [sng_parser.parse(cap) for cap in caps]
            for i, g in enumerate(raw_graphs):
                entities = g["entities"]
                relations = g["relations"]
                counter_entity.update([e["lemma_head"] for e in entities])
                counter_relation.update([r["lemma_relation"] for r in relations])
                # TODO find out the logic he used for Stop words or proper name
        vocab_entity = tt.vocab.Vocab(counter_entity, min_freq=freq)
        torch.save(vocab_entity, ent )
        vocab_relation = tt.vocab.Vocab(counter_relation, min_freq=freq)
        torch.save(vocab_relation, rel )
    else:
        print("Loading Vocabulary.")
        vocab_entity = torch.load(ent)
        vocab_relation = torch.load(rel)
    return vocab_entity, vocab_relation


def extract_text_graph(all_caps, entity_vocabulary, relation_vocabulary):
    """

    :param all_caps:
    :param entity_vocabulary:
    :param relation_vocabulary:
    :return:
    """

    new_graphs = {}
    for k in tqdm(all_caps.keys()):
        caps = all_caps[k]
        raw_graphs = [sng_parser.parse(cap) for cap in caps]
        cleaned_graphs = []
        for i, g in enumerate(raw_graphs):
            entities = g["entities"]
            relations = g["relations"]
            # print(str(i),"\n")
            # print(caps[i])
            # print("\n")
            # print(graphs[i])
            # if len (entities) == 0 or len (relations) == 0:
            #     continue
            # else:
            # TODO find out the logic he used for Stop words or proper name
            filtered_entities = [e["lemma_head"] if e["lemma_head"] in entity_vocabulary else 'none' for e in entities]
            filtered_relations = [[r["subject"], r["object"], r["lemma_relation"]] for r in relations if
                                  r["lemma_relation"] in relation_vocabulary]
            extracted_graph = {'entities': filtered_entities, 'relations': filtered_relations}
            cleaned_graphs.append(extracted_graph)

        new_graphs[k] = cleaned_graphs

    return new_graphs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform Normal Sentences to Graphs")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    parser.add_argument(
        "--outpath",
        default="/media/rafi/Samsung_T5/_DATASETS/",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    parser.add_argument(
        "--graph_file_name",
        default="new_graphs.json",
        metavar="FILE",
        help="file_name.json",
        type=str,
    )

    args = parser.parse_args()
    spacy.load('en_core_web_sm')

    ent_file = "vocab_entity.pth"
    rel_file = "vocab_relation.pth"
    data_dir = DatasetCatalog.DATA_DIR
    attrs = DatasetCatalog.DATASETS["VG_stanford_filtered_with_attribute"]
    cap_graph_file = os.path.join(data_dir, attrs["capgraphs_file"])
    vg_dict_file = os.path.join(data_dir, attrs["dict_file"])
    image_file = os.path.join(data_dir, attrs["image_file"])
    roidb_file = os.path.join(data_dir, attrs["roidb_file"])

    cap_graph = json.load(open(cap_graph_file))
    vg_dict = json.load(open(vg_dict_file))

    all_graphs = cap_graph["vg_coco_id_to_capgraphs"]
    all_caps = cap_graph["vg_coco_id_to_caps"]

    entity_vocabulary = cap_graph["cap_category"].keys()
    relation_vocabulary = cap_graph["cap_predicate"].keys()

    entity_vocab, relation_vocab = make_vocab(all_caps, args.outpath, ent_file, rel_file)
    news_graphs = extract_text_graph(all_caps, entity_vocabulary, relation_vocabulary)

    with open(os.path.join(args.outpath, args.graph_file_name), 'w', encoding='utf-8') as f:
        json.dump(news_graphs, f, ensure_ascii=False, indent=4)
    print("Saved graph")
    # Looks like "this" is a stop word => completely removed.
    # Proper Name are replaced by 'none'
