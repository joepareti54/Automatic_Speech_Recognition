import json
import os 
import librosa
path  =  '/home/joseph/ASRVerify'
path_t=  '/home/joseph/ASRVerify'

#set parameters for training and checkpointing files
START_LINE = 40000
END_LINE = 41000
#
NUMB_OF_LINES = END_LINE - START_LINE
NUMB_OF_VALID_LINES = 100
MAX_EPOCHS = 50 

OUT_DIR = '/home/joseph/ASRVerify/'
#OUT_FILE=OUT_DIR+'out_file.txt'
TRAIN_MANIFEST=OUT_DIR+'Vmanifest.json'
TEST_MANIFEST=OUT_DIR+'Vt-manifest.json'


import nemo
import nemo.collections.asr as nemo_asr

try:
    from ruamel.yaml import YAML
except ModuleNotFoundError:
    from ruamel_yaml import YAML
config_path = '/home/joseph/ASRVerify/config.yaml'

yaml = YAML(typ='safe')
with open(config_path) as f:
    params = yaml.load(f)
print(params)


# ### Training with PyTorch Lightning
import pytorch_lightning as pl
trainer = pl.Trainer(gpus=0, max_epochs=MAX_EPOCHS)
CHECKPOINT_DIR = OUT_DIR + 'CHECKPOINT/'
CHECKPOINT_FILE_OUT = CHECKPOINT_DIR + 'train-' + str(START_LINE) + '-' + str(END_LINE)+'-valid-End-'+str(NUMB_OF_VALID_LINES)+'-'+str(MAX_EPOCHS)+'e.chk'
#
import glob 
list_of_files = glob.glob(CHECKPOINT_DIR +'*') # * means all if need specific format then *.csv
CHECKPOINT_FILE_IN = max(list_of_files, key=os.path.getctime)
#
print('input checkpoint file ',CHECKPOINT_FILE_IN)

#
from omegaconf import DictConfig
params['model']['train_ds']['manifest_filepath'] = TRAIN_MANIFEST
params['model']['validation_ds']['manifest_filepath'] = TEST_MANIFEST
MODEL = nemo_asr.models.EncDecCTCModel.restore_from(CHECKPOINT_FILE_IN)
MODEL.setup_training_data(train_data_config=params['model']['train_ds'])
MODEL.setup_validation_data(val_data_config=params['model']['validation_ds'])
trainer.fit(MODEL)

MODEL.save_to(CHECKPOINT_FILE_OUT)


print(MODEL.transcribe(paths2audio_files=[path+'/p247_010.wav', path+'/p279_217.wav', path+'/p279_218.wav', path+'/p279_219.wav'],
                                                    
                                 batch_size=4))

