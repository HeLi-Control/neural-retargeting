from yacs.config import CfgNode as CN

_C = CN()

_C.DATASET = CN()
_C.DATASET.TYPE = None
_C.DATASET.TRAIN = CN()
_C.DATASET.TRAIN.SOURCE_NAME = None
_C.DATASET.TRAIN.SOURCE_PATH = None
_C.DATASET.TRAIN.TARGET_NAME = None
_C.DATASET.TRAIN.TARGET_PATH = None
_C.DATASET.TEST = CN()
_C.DATASET.TEST.SOURCE_NAME = None
_C.DATASET.TEST.SOURCE_PATH = None
_C.DATASET.TEST.TARGET_NAME = None
_C.DATASET.TEST.TARGET_PATH = None

_C.TRAIN = CN()
_C.TRAIN.READ_SPECIFIC_YAML = None
_C.TRAIN.SPECIFIC_YAML_NAME = None

_C.MODEL = CN()
_C.MODEL.NAME = None
_C.MODEL.LOAD_CHECKPOINT = None
_C.MODEL.CHECKPOINT = None
_C.MODEL.CHECKPOINT_RANDOM = None

_C.HYPER = CN()
_C.HYPER.TRAIN_TIMES = None
_C.HYPER.EPOCHS = None
_C.HYPER.BATCH_SIZE = None
_C.HYPER.LEARNING_RATE = None
_C.HYPER.VARIABLE_LEARNING_RATE = None
_C.HYPER.VARIABLE_LEARNING_RATE_GAIN = None

_C.LOSS = CN()
_C.LOSS.EE = False
_C.LOSS.VEC = False
_C.LOSS.COL = False
_C.LOSS.COL_THRESHOLD = None
_C.LOSS.LIM = False
_C.LOSS.ORI = False
_C.LOSS.ORI_LOSS_CALC_METHOD = None
_C.LOSS.FIN = False
_C.LOSS.REG = False
_C.LOSS.LOSS_USING_GAIN = False
_C.LOSS.LOSS_GAIN = None

_C.INFERENCE = CN()
_C.INFERENCE.H5 = CN()
_C.INFERENCE.H5.BOOL = None
_C.INFERENCE.H5.PATH = None
_C.INFERENCE.PYBULLET = CN()
_C.INFERENCE.PYBULLET.BOOL = None
_C.INFERENCE.RUN = CN()
_C.INFERENCE.RUN.INFERENCE = None
_C.INFERENCE.RUN.INFERENCE_LOSS = None
_C.INFERENCE.RUN.HUMAN_DEMONSTRATE = None

_C.OTHERS = CN()
_C.OTHERS.SAVE = None
_C.OTHERS.SUMMARY = None
_C.OTHERS.LOG = None
_C.OTHERS.LOG_INTERVAL = None

cfg = _C
