from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

_CC = _C

# ----------- Backbone ----------- #
_CC.MODEL.BACKBONE.FREEZE = False
_CC.MODEL.BACKBONE.FREEZE_AT = 3

# ----------- GLMD ----------- #
_CC.MODEL.ENABLE_GLMD = False
_CC.MODEL.GLMD = CN()
_CC.MODEL.GLMD.ENABLE_CM = False
_CC.MODEL.GLMD.ENABLE_LR = False
_CC.MODEL.GLMD.ENABLE_MDF = False
_CC.MODEL.GLMD.ENABLE_CRD = False

# ------------- RPN -------------- #
_CC.MODEL.RPN.FREEZE = False
_CC.MODEL.RPN.ENABLE_DECOUPLE = False
_CC.MODEL.RPN.BACKWARD_SCALE = 1.0
_CC.MODEL.RPN.BBOX_OBJ_LOSS_TYPE = "normal"

# ------------- ROI -------------- #
_CC.MODEL.ROI_HEADS.NAME = "Res5ROIHeads"
_CC.MODEL.ROI_HEADS.FREEZE_FEAT = False
_CC.MODEL.ROI_HEADS.ENABLE_DECOUPLE = False
_CC.MODEL.ROI_HEADS.BACKWARD_SCALE = 1.0

_CC.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers"
# scale of cosine similarity (set to -1 for learnable scale)
_CC.MODEL.ROI_HEADS.COSINE_SCALE = 20.0
_CC.MODEL.ROI_HEADS.CLS_DROPOUT = False
_CC.MODEL.ROI_HEADS.DROPOUT_RATIO = 0.8

_CC.MODEL.ROI_BOX_HEAD.BBOX_CLS_LOSS_TYPE = "CE"
_CC.MODEL.ROI_BOX_HEAD.DECOUPLE_CLS_REG = False
_CC.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7  # for faster

# ------------- TEST ------------- #
_CC.TEST.PCB_ENABLE = False
_CC.TEST.PCB_MODELTYPE = 'resnet'             # res-like
_CC.TEST.PCB_MODELPATH = ""
_CC.TEST.PCB_ALPHA = 0.50
_CC.TEST.PCB_UPPER = 1.0
_CC.TEST.PCB_LOWER = 0.05

# ------------ Other ------------- #
_CC.SOLVER.WEIGHT_DECAY = 5e-5
_CC.MUTE_HEADER = True
