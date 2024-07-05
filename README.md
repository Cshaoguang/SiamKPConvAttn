# SiamKPConvAttn

## Installation

```
# It is recommended to create a new environment
conda create -n siamkpconvattn python==3.8
conda activate siamkpconvattn

# [Optional] If you are using CUDA 11.0 or newer, please install `torch==1.7.1+cu110`
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# Install packages and other dependencies
pip install -r requirements.txt
pip install "laspy[lazrs,laszip]"
python setup.py build develop 

```

## Data Preparation
```
 ## SiamKPConvAttn
  ## Synthetic
 
The data should be organized as follows:
--time_a
       |--test
       |--train
       |--val
--time_b
       |--test
       |--train
       |--val
--labeled_point_lists_syn
                        |--test
                        |--train
                        |--val
```
## Training

```
#Set the Python path to include the parent folder by running the following command:
export PYTHONPATH=..:$PYTHONPATH
     ## SiamKPConvAttn
      ## Synthetic
          #Run the command:
          python experiments/synthetic/trainval.py
```

## Testing

```
#The test data needs to be placed in the 'test' folder of each dataset.
     ## SiamKPConvAttn
      ## Synthetic
          #Run the command:
          python experiments/synthetic/test.py --snapshot best_model.pth.tar
```
