# CNTdocker

## About 
Dockerfiles to create Docker images used by the CNT at the university of Pennsylvania

## Directory contents explanation

### **EEG**: 
  Dockerfiles used to create images with common EEG analysis tools. Usually python 3
  
  **echobase**: Dockerfiles used to create images that can calculate functional connectivity of EEG
    Also has ieegpy python package used to interface with iEEG.org
    Echobase code is from https://github.com/andyrevell/paper001
    
    Ubuntu 18.04
      Python 2.7 and Python 3.6
      Numpy 1.18.4
      pandas 1.0.3
       scipy 1.4.1
       
 ### **Imaging**: 
  Dockerfiles used to create images with common MRI analysis tools.
    
      Ubuntu 18.04
      Python 2.7, Python 3.6, Python 3.7
      dcm2niix
      dsistudio
      ANTS
      Freesurfer
      FSL 6.0.1
      
### **ml**: 
  Dockerfiles used to create images with common machine learning tools.
  
  **wavenet**: Dockerfile to create compatible dependencies to use with Goodgle Deepmind wavenet paper
    [Wavenet blog](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)
    [Wavenet paper](https://arxiv.org/pdf/1609.03499.pdf)
    
      Ubuntu 18.04
      tensorflow 1.0.0
      pandas 0.19.2
      librosa 0.5.0
      
  **Tensorflow_2.1**: Dockerfile to create compatible dependencies to with tensorflow 2.1
      
      Ubuntu 18.04
      tensorflow 2.1
  
     
