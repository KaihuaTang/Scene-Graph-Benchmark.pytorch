CUDA_VISIBLE_DEVICES=0,1 

python -m torch.distributed.launch \
--master_port 10025 \
--nproc_per_node=2 \
tools/relation_train_net.py \
--config-file configs/e2e_relation_X_101_32_8_FPN_1x.yaml \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
SOLVER.IMS_PER_BATCH 12 \
TEST.IMS_PER_BATCH 2 \
DTYPE float16 \
SOLVER.MAX_ITER 50000 \
SOLVER.VAL_PERIOD 2000 \
SOLVER.CHECKPOINT_PERIOD 10000 \
SOLVER.PRE_VAL False \
GLOVE_DIR /home/luoc/workspace/scene-graph/glove \
MODEL.PRETRAINED_DETECTOR_CKPT /home/luoc/workspace/scene-graph/checkpoints/pretrained_faster_rcnn/model_final.pth \
OUTPUT_DIR /home/luoc/workspace/scene-graph/checkpoints/motif-precls-test1
