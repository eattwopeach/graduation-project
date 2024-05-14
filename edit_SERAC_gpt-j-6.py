import os

from easyeditor import BaseEditor
from easyeditor import KNHyperParams, FTHyperParams, KETrainingHparams,\
    ROMEHyperParams, MEMITHyperParams, MENDTrainingHparams, MENDHyperParams, \
    SERACTrainingHparams, SERACHparams, IKEHyperParams, FTApiHyperParams, LoRAHyperParams, \
    GraceHyperParams
from easyeditor import ZsreDataset, CounterFactDataset
from easyeditor import EditTrainer

## Loading config from hparams/MEMIT/gpt2-xl.yaml
#hparams = ROMEHyperParams.from_hparams('/home/u200111312/EasyEdit/hparams/ROME/chatglm2-6b.yaml')
hparams = SERACTrainingHparams.from_hparams('hparams/SERAC/gpt-j-6B.yaml')

