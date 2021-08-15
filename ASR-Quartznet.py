import json
import os 
import librosa
path  =  '/home/joseph/ASRVerify'
path_t=  '/home/joseph/ASRVerify'


import nemo
import nemo.collections.asr as nemo_asr
quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
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

print(quartznet.transcribe(paths2audio_files=[path+'/p247_010.wav', path+'/p279_217.wav', path+'/p279_218.wav', path+'/p279_219.wav'],
                                                    
                                 batch_size=4))

