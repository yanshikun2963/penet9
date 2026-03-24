#!/bin/bash
# R1: PredCls vanilla (IMS_PER_BATCH=8, matching original penet config)
# NOTE: IMS_PER_BATCH is used as rl_factor to scale learning rate.
#       batch=8 -> actual lr = BASE_LR * 8 = 0.001 * 8 = 0.008
#       batch=12 was WRONG -> actual lr = 0.001 * 12 = 0.012 (50% too high!)
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export CB_BETA=none

MODEL_NAME='R1_PredCls_vanilla'
mkdir -p ./checkpoints/${MODEL_NAME}/

python3 -u \
  tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 \
  SOLVER.MAX_ITER 36000 SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(16000, 28000)" SOLVER.VAL_PERIOD 3000 \
  SOLVER.CHECKPOINT_PERIOD 6000 GLOVE_DIR ./datasets/vg/ \
  MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0 \
  INPUT.MIN_SIZE_TRAIN "(600,)" \
  INPUT.MAX_SIZE_TRAIN 1000 \
  2>&1 | tee ./checkpoints/${MODEL_NAME}/train.log
