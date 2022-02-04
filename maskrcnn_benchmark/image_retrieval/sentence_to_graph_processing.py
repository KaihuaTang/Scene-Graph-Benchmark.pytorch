import argparse
import os
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
import sng_parser
import json
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform Normal Sentences to Graphs")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    args = parser.parse_args()

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


    # Looks like "this" is a stop word => completely removed.
    # Proper Name are replaced by 'none'
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
            #TODO find out the logic he used for Stop words or proper name
            filtered_entities = [ e["lemma_head"] if e["lemma_head"] in entity_vocabulary else 'none'  for e in entities  ]
            filtered_relations = [ [ r["subject"], r["object"], r["lemma_relation"] ]  for r  in relations if r["lemma_relation"] in relation_vocabulary]
            extracted_graph = {'entities': filtered_entities, 'relations': filtered_relations}
            cleaned_graphs.append(extracted_graph)

        new_graphs[k] = cleaned_graphs



    pass