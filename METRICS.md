# Explanation of our metrics
### Recall@K (R@K)
The earliest and the most widely accepted metric in scene graph generation, which is firstly adopted by [Visual relationship detection with language priors](https://arxiv.org/abs/1608.00187). Since the ground-truth annotations of relationships are incomplete, it's improper to use simple accurary as the metric. Therefore, Lu et al. transfer it to a retrieve-like problem: the relationships are not only required to be correctly classified, but also required to have as higher score as possible, so they can be retrieved from plenty of 'none' relationship pairs.

### No Graph Constraint Recall@K (ng-R@K)
It's firstly used by [Pixel2Graph](https://arxiv.org/abs/1706.07365) and named by [Neural-MOTIFS](https://arxiv.org/abs/1711.06640). The former paper significantly improves the R@K results by allowing each pair to have multiple predicates, which means for each subject-object pair, all the 50 predicates will be involved in the recall ranking not just the one with highest score. Since predicates are not exclusive, 'on' and 'riding' can both be correct. This setting significantly improves the R@K. To fairly compare with other methods, [Neural-MOTIFS](https://arxiv.org/abs/1711.06640) named it as the No Graph Constraint Recall@K (ngR@K).

### Mean Recall@K (mR@K)
It is proposed by our work [VCTree](https://arxiv.org/abs/1812.01880) and Chen et al.s'[KERN](https://arxiv.org/abs/1903.03326) at the same time (CVPR 2019), although we didn't make it as our main contribution and only listed the full results on the [supplementary material](https://zpascal.net/cvpr2019/Tang_Learning_to_Compose_CVPR_2019_supplemental.pdf). However, we also acknowledge the contribution of [KERN](https://arxiv.org/abs/1903.03326), for they gave more mR@K results of previous methods. The main motivation of Mean Recall@K (mR@K) is that the VisualGenome dataset is biased towards dominant predicates. If the 10 most frequent predicates are correctly classified, the accuracy would reach 90% even the rest 40 kinds of predicates are all wrong. This is definitely not what we want. Therefore, Mean Recall@K (mR@K) calculates Recall@K for each predicate category independently then report their mean. 

### No Graph Constraint Mean Recall@K (ng-mR@K)
The same mean Recall metric, but for each pair of objects, all possible predicates are valid candidates (the original mean Recall@K only considers the predicate with maximum score of each pair as the valid candidate to calculate Recall).

### Zero Shot Recall@K (zR@K)
It is firstly used by [Visual relationship detection with language priors](https://arxiv.org/abs/1608.00187) for VRD dataset, and firstly reported by  [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949) for VisualGenome dataset. In short, it only calculates the Recall@K for those subject-predicate-object combinations that not occurred in the training set.

### No Graph Constraint Zero Shot Recall@K (ng-zR@K)
The same zero-shot Recall metric, but for each pair of objects, all possible predicates are valid candidates (the original zero-shot Recall@K only considers the predicate with maximum score of each pair as the valid candidate to calculate Recall).

### Top@K Accuracy (A@K) 
It is actually caused by the misunderstanding of PredCls and SGCls protocols. [Contrastive Losses](https://arxiv.org/abs/1903.02728) reported Recall@K of PredCls and SGCls by not just giving ground-truth bounding boxes, but also giving the ground-truth subject-object pairs, so no ranking is involved. The results can only be considerred as Top@K Accuracy (A@K) for the given K ground-truth subject-object pairs. 

### Sentence-to-Graph Retrieval (S2G)
S2G is proposed by [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949) as an ideal downstream task that only relies on the quality of SGs, for the existing VQA and Captioning are too complicated and challenged by their own bias. It takes human descriptions as queries, searching for matching scene graphs (images), where SGs are considered as the symbolic representations of images. More details will be explained in [S2G-RETRIEVAL.md](maskrcnn_benchmark/image_retrieval/S2G-RETRIEVAL.md).

# Two Common Misunderstandings in SGG Metrics
When you read/follow a SGG paper, and you find that its performance is abnormally high for no obvious reasons, whose authors could mess up some metrics.

1. Not differentiate Graph Constraint Recall@K and No Graph Constraint Recall@K. The setting of With/Without Constraint is introduced by [Neural-MOTIFS](https://arxiv.org/abs/1711.06640). However, some early work and a few recent researchers don't differentiate these two setting, using No Graph Constraint results to compare with previous work With Graph Constraint. TYPICAL SYMPTOMS: 1) Recall@100 of PredCls is larger than 75%, 2) not mention With/Without Graph Constraint in the original paper. TYPICAL Paper:[Pixel2Graph](https://arxiv.org/abs/1706.07365) (Since this paper is published before MOTIFS, they didn't mean to take this advantage, and they are actually the fathers of No Graph Constraint setting while MOTIFS is the one who named this baby.)

2. Some researchers misunderstand the protocols of PredCls and SGCls. These two protocols only give ground-truth bounding boxes NOT ground-truth subject object pairs. Some works only predict relationships for ground-truth subject-object pairs in PredCls and SGCls, so their PredCls and SGCls results will extremely high. Note that Recall@K metric is a ranking metric, using ground-truth subject-object pairs can be considerred as giving the perfect ranking. In order to separate from normal PredCls and SGCls,  I name this kind of setting as Top@K Accuracy, which is only applicable to PredCls and SGCls. TYPICAL SYMPTOMS: 1) results of PredCls and SGCls are extremely high while results of SGGen are normal, 2) Recall@50 and Recall@100 of PredCls and SGCls are exactly the same, since the ranking is perfect (Recall@20 is less, for some images have groud-truth relationships more than 20). TYPICAL Paper:[Contrastive Losses](https://arxiv.org/abs/1903.02728).

# Output Format of Our Code

![alt text](demo/output_format.png "from 'screenshot'")


# Reported Results

### The results of reimplemented [IMP](https://arxiv.org/abs/1701.02426), [MOTIFS](https://arxiv.org/abs/1711.06640), [VCTree](https://arxiv.org/abs/1812.01880) and our Transformer with X-101-FPN backbone
All the following results only used two 1080ti GPUs with batch size 8 (increasing batch size will further improves some models a little bit). Note that the reimplemented VCTree is not exactly the same as the [original work](https://github.com/KaihuaTang/VCTree-Scene-Graph-Generation). It's an optimized version for SGCls and SGGen. But PredCls seems not as good as previous, I will try to find the reason later. Hybrid Learning is discarded for simplicity.

### Recall@K

Models | SGGen R@20 | SGGen R@50 | SGGen R@100 | SGCls R@20 | SGCls R@50 | SGCls R@100 | PredCls R@20 | PredCls R@50 | PredCls R@100
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
IMP | 18.09 | 25.94 | 31.15 | 34.01 | 37.48 | 38.50 | 54.34 | 61.05 | 63.06
MOTIFS | 25.48 | 32.78 | 37.16 | 35.63 | 38.92 | 39.77 | 58.46 | 65.18 | 67.01
Transformer | 25.55 | 33.04 | 37.40 | 36.87 | 40.18 | 41.02 | 59.06 | 65.55 | 67.29
VCTree | 24.53 | 31.93 | 36.21 | 42.77 | 46.67 | 47.64 | 59.02 | 65.42 | 67.18

### No Graph Constraint Recall@K

Models | SGGen ngR@20 | SGGen ngR@50 | SGGen ngR@100 | SGCls ngR@20 | SGCls ngR@50 | SGCls ngR@100 | PredCls ngR@20 | PredCls ngR@50 | PredCls ngR@100
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
IMP | 18.35 | 27.02 | 33.89 | 38.70 | 46.78 | 51.20 | 62.14 | 76.82 | 84.97
MOTIFS | 27.04 | 36.58 | 43.43 | 40.58 | 48.48 | 51.98 | 66.39 | 81.02 | 88.24
Transformer | 27.14 | 36.98 | 43.90 | 42.31 | 50.18 | 53.93 | 67.45 | 81.83 | 88.95
VCTree | 26.14 | 35.73 | 42.34 | 48.94 | 58.36 | 62.70 | 67.20 | 81.63 | 88.83

### Zero Shot Recall@K 
Note: IMP achieves highest Zero Shot Recall@K because it doesn't include any explicit or implicit object label embeddings for predicate prediction.

Models | SGGen zR@20 | SGGen zR@50 | SGGen zR@100 | SGCls zR@20 | SGCls zR@50 | SGCls zR@100 | PredCls zR@20 | PredCls zR@50 | PredCls zR@100
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
IMP | 0.18 | 0.38 | 0.77 | 2.01 | 3.30 | 3.92 | 12.17 | 17.66 | 20.25
MOTIFS | 0.0 | 0.05 | 0.11 | 0.32 | 0.68 | 1.13 | 1.08 | 3.24 | 5.36
Transformer | 0.04 | 0.14 | 0.29 | 0.34 | 0.91 | 1.39 | 1.35 | 3.63 | 5.64
VCTree | 0.10 | 0.31 | 0.69 | 0.45 | 1.17 | 2.08| 1.04 | 3.27 | 5.51

### Mean Recall@K 

Models | SGGen mR@20 | SGGen mR@50 | SGGen mR@100 | SGCls mR@20 | SGCls mR@50 | SGCls mR@100 | PredCls mR@20 | PredCls mR@50 | PredCls mR@100
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
IMP | 2.75 | 4.17 | 5.30 | 5.21 | 6.18 | 6.53 | 8.85 | 10.97 | 11.77
MOTIFS | 4.98 | 6.75 | 7.90 | 6.68 | 8.28 | 8.81 | 11.67 | 14.79 | 16.08
Transformer | 6.01 | 8.13 | 9.56 | 8.14 | 10.09 | 10.73 | 12.77 | 16.30 | 17.63
VCTree | 5.38 | 7.44 | 8.66 | 9.59 | 11.81 | 12.52 | 13.12 | 16.74 | 18.16

### Top@K Accuracy

Models | SGGen A@20 | SGGen A@50 | SGGen A@100 | SGCls A@20 | SGCls A@50 | SGCls A@100 | PredCls A@20 | PredCls A@50 | PredCls A@100
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
IMP | - | - | - | 39.19 | 39.30 | 39.30 | 64.88 | 65.12 | 65.12
MOTIFS | - | - | - | 40.41 | 40.50 | 40.50 | 68.87 | 69.14 | 69.14
Transformer | - | - | - | 41.75 | 41.84 | 41.84 | 69.08 | 69.36 | 69.36
VCTree | - | - | - | 48.47 | 48.59 | 48.59 | 68.92 | 69.19 | 69.19

### The results of [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949) with X-101-FPN backbone

Note that if you are using the default VCTree settings of this project, all results of VCTree should be better than what we reported in [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949), i.e., the following results, because we optimized the tree construction network after the publication.

### Recall@K and Mean Recall@K 

![alt text](demo/TDE_Results1.png "from 'Unbiased Scene Graph Generation from Biased Training'")

### Zero Shot Recall@K

![alt text](demo/TDE_Results2.png "from 'Unbiased Scene Graph Generation from Biased Training'")