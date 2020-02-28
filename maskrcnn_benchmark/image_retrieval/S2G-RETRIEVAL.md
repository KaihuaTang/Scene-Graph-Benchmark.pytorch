# Sentence-to-Graph Retrieval (S2G)

Forgive me, this part of code is ugly and less organized.

## Preprocessing

Run the ```maskrcnn_benchmark/image_retrieval/preprocessing.py``` to process the annotations and checkpoints, where ```detected_path``` should be set to the corresponding checkpoints you want to use, ```vg_data, vg_dict, vg_info``` should have already downloaded if you followed DATASET.md, ```cap_graph``` is the ground-truth captions and generated sentence graphs (you can download it from [here](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779999&authkey=AGW0Wxjb1JSDFnc)). We use [SceneGraphParser](https://github.com/vacancy/SceneGraphParser) to generate these sentence graphs.

You also need to set the ```cap_graph``` PATH and ```vg_dict``` PATH in ```maskrcnn_benchmark/image_retrieval/dataloader.py``` manually.

## Training and Evaluation

Run the ```tools/image_retrieval_main.py``` for both training and evaluation. 

To load the generated scene graphs of the given SGG checkpoints, you need to manually set ```sg_train_path``` and ```sg_test_path``` in ```tools/image_retrieval_main.py```, which means you need to evaluate your model on **both training and testing set** to get the generated crude scene graphs. Our evaluation code will automatically saves the crude SGGs into ```checkpoints/MODEL_NAME/inference/VG_stanford_filtered_wth_attribute_test/```  or ```checkpoints/MODEL_NAME/inference/VG_stanford_filtered_wth_attribute_train/```, which will be further processed to generate the input of ```sg_train_path``` and ```sg_test_path``` by our preprocessing code ```maskrcnn_benchmark/image_retrieval/preprocessing.py```.

## Results

Sentence-to-Graph Retrieval (S2G) results are given in the paper [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949):

![alt text](../../demo/TDE_Results3.png "from 'Unbiased Scene Graph Generation from Biased Training'")
