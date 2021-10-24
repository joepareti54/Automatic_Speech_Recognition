# Automatic_Speech_Recognition

This work is based on code provided within Nvidia NeMo: https://github.com/NVIDIA/NeMo/blob/v1.0.2/tutorials/asr/01_ASR_with_NeMo.ipynb

The notebook NB_All.ipynb trains the EncDecCTCModel out of Nvidia NeMo using the entire VCTK dataset corpus 

The Python program ASR_Train_Inf.py reads a checkpointed  EncDecCTCModel and does inferences; it also trains the model on a specified segment of 
the VCTK data coprpus.

The Python program ASR-Quartznet.py downloads QuartzNet out of Nvidia NeMo and does inferences using some records from the VCTK corpus. 

Some important development notes are here https://github.com/NVIDIA/NeMo/issues/2628

The results are here: https://docs.google.com/spreadsheets/d/1Dg6NlH_BR1aCqpLUMA8I1cz5GCIkLpACNdXUj_hgR_Y/edit

Conclusions:
I achieved the best results using QuartzNet pre-trained model even though it never saw my VCTK corpus dataset. 

The QuartzNet model was trained on nearly 5000 hours of speech, roughly 2.5 Million audio files, from various public corpora. 
Fine Tuning on a specific domain will usually help that domain only and degrade generalization performance on other domains 
unless the corpus is as large as or even larger than the original dataset.

But my VCTK corpus is much smaller therefore there will be no further performance improvement.

Additional Project Details:
- Based on NeMo: https://docs.google.com/document/d/1bZQc1GIQ3oQHSXMoRi5bOBF4jcK_oEEElJ7WUPvrmZ0/edit#
- Based on Listen Attend Spell models : https://docs.google.com/document/d/1jkljv9BlOkVwP7E78EpSh7nSiYqrY2tXXyWOduPD74g/edit#

# special note for myself, in order to keep track on what was done
The programs have been run from my HP laptop which is cpu only. This proves that a GPU is NOT mandatory for inference and testing
