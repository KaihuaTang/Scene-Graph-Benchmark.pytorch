import argparse
import os
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
import sng_parser
import json
from tqdm import tqdm
import collections
import torch
import torchtext as tt
import pandas

def make_vocab(all_caps, outpath, file_name_entity, file_name_relation, freq=5):
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


def extract_text_graph_no_stopwords(all_caps, entity_vocabulary, relation_vocabulary):
    """
    This function tries to recreate the data found under "vg_coco_id_to_capgraphs", in the file vg_capgraphs_anno.json.
    Only based on the existing vocabulary that was extracted from vg_capgraphs_anno.json.
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
            filtered_entities = [e["lemma_head"] if e["lemma_head"] in entity_vocabulary else 'none' for e in entities]
            filtered_relations = [[r["subject"], r["object"], r["lemma_relation"]] for r in relations if
                                  r["lemma_relation"] in relation_vocabulary]

            extracted_graph = {'entities': filtered_entities, 'relations': filtered_relations}
            cleaned_graphs.append(extracted_graph)

        new_graphs[k] = cleaned_graphs

    return new_graphs

def extract_text_graph(all_caps, entity_vocabulary, relation_vocabulary, stop_words):
    """
    This function tries to recreate the data found under "vg_coco_id_to_capgraphs", in the file vg_capgraphs_anno.json.
    Only based on the existing vocabulary that was extracted from vg_capgraphs_anno.json, and a list of stop words if
    needed.
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
            filtered_entities = []
            stop_i = []
            for i , e in enumerate(entities):
                entity_lemma = e["lemma_head"]
                if entity_lemma in entity_vocabulary:
                    filtered_entities.append(entity_lemma)
                elif entity_lemma in stop_words:
                    #filtered_entities.append('STOP')
                    stop_i.append(i)
                else:
                    filtered_entities.append('none')
            pass
            # filtered_entities = [e["lemma_head"] if e["lemma_head"] in entity_vocabulary else 'none' for e in entities]
            filtered_relations = [[r["subject"], r["object"], r["lemma_relation"]] for r in relations if
                                  r["lemma_relation"] in relation_vocabulary]

            # remove stop words and adjust index for relations
            for s in reversed(stop_i):
                last_index = len(filtered_relations) - 1
                for i in range(last_index, -1, -1):
                    first_index = filtered_relations[i][0]
                    second_index = filtered_relations[i][1]
                    if first_index > s:
                        filtered_relations[i][0] = first_index - 1
                    if second_index > s:
                        filtered_relations[i][1] = second_index - 1
                    if first_index == s or second_index == s:
                        filtered_relations.pop(i)

            extracted_graph = {'entities': filtered_entities, 'relations': filtered_relations}
            cleaned_graphs.append(extracted_graph)

        new_graphs[k] = cleaned_graphs

    return new_graphs

def check_graph_versus_original(original_graphs, new_graphs):


    different_list = {}
    for k in tqdm(original_graphs.keys()):
        original = original_graphs[k]
        new = new_graphs[k]
        count = 0
        count_rel = 0
        for i, _ in enumerate(original):
            if len(original[i]["entities"]) != len(new[i]["entities"]) :
                count += 1
            if len(original[i]["relations"]) != len(new[i]["relations"]):
                count_rel += 1
        if count > 0 or count_rel > 0:
            different_list[k] = {"ent_diff":count, "rel_diff":count_rel }
    return different_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform Normal Sentences to Graphs")

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
    ent_file = "vocab_entity.pth"
    rel_file = "vocab_relation.pth"
    data_dir = DatasetCatalog.DATA_DIR
    attrs = DatasetCatalog.DATASETS["VG_stanford_filtered_with_attribute"]
    # This is vg_capgraphs_anno.json
    cap_graph_file = os.path.join(data_dir, attrs["capgraphs_file"])
    vg_dict_file = os.path.join(data_dir, attrs["dict_file"])


    cap_graph = json.load(open(cap_graph_file))
    vg_dict = json.load(open(vg_dict_file))

    all_graphs = cap_graph["vg_coco_id_to_capgraphs"]
    all_caps = cap_graph["vg_coco_id_to_caps"]

    entity_vocabulary = cap_graph["cap_category"].keys()
    relation_vocabulary = cap_graph["cap_predicate"].keys()
    save_path = os.path.join(args.outpath, args.graph_file_name)

    entity_vocab, relation_vocab = make_vocab(all_caps, args.outpath, ent_file, rel_file)

    if os.path.exists(save_path):
        new_graphs = json.load(open(save_path))
    else:
        # stop_words = [e for e in entity_vocab.itos if e not in entity_vocabulary]
        # new_graphs = extract_text_graph(all_caps, entity_vocabulary, relation_vocabulary, stop_words)
        new_graphs = extract_text_graph_no_stopwords(all_caps, entity_vocabulary, relation_vocabulary)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(new_graphs, f, ensure_ascii=False, indent=4)
        print("Saved graph")

    # diff_list = check_graph_versus_original(all_graphs, new_graphs)
    # df = pandas.DataFrame([diff_list[k] for k in diff_list.keys()])
    # print(df.mean())
