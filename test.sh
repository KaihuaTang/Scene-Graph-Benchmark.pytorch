CUDA_VISIBLE_DEVICES=0,1

python -m torch.distributed.launch \
--master_port 10028 \
--nproc_per_node=2 \
tools/relation_test_net.py \
--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor \
MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE \
MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum \
MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  \
TEST.IMS_PER_BATCH 2 \
DTYPE "float16" \
GLOVE_DIR /home/luoc/workspace/scene-graph/glove \
MODEL.PRETRAINED_DETECTOR_CKPT /home/luoc/workspace/scene-graph/checkpoints/causal_motifs_sgdet \
OUTPUT_DIR /home/luoc/workspace/scene-graph/checkpoints/causal_motifs_sgdet