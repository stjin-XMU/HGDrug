# HGDrug
The code is an official PyTorch-based implementation in the paper “A General Hypergraph Learning Framework for Drug Multi-task Predictions”
# Abstract
The powerful combination of large-scale drug-related interaction networks and deep learning provides new opportunities for drug discovery and drug repositioning. However, chemical structure that play important role in drug properties is neglected in current biomedical networks. Here we present a general hypergraph learning framework, which introduces Drug-Substructures relationship into Molecular interaction  Networks to construct a micro-to-macro drug centric heterogeneous network (DSMN), and develops a multi-branches  Hyper Graph learning model, called HGDrug, for Drug multi-task predictions. The HGDrug framework is designed to capture high-order drug relationships and obtain effective drug features from DSMN network by motif-driven hypergraphs and self-supervised auxiliary task. HGDrug achieves high accuracy and robust predictions on 4 benchmark tasks (drug-drug, drug-target, drug-disease, and drug-side-effect interactions), outperforming all 6 general-purpose classical models and 8 state-of-the-art task-specific models. Experiments analysis verify the effectiveness and rationality of the model architecture and multi-branches setup, and demonstrated HGDrug can capture the approximate relationship between drugs with the same functional group. More importantly, the constructed drug-substructure interaction networks can help improve the performance of existing network models for drug-related interactions prediction tasks. The code of our model is available via https://github.com/stjin-XMU/HGDrug.
![DSMN construction](https://github.com/stjin-XMU/HGDrug/blob/main/HGDrug_1.png)

![HGDrug model](https://github.com/stjin-XMU/HGDrug/blob/main/2023HGDrug.png)
## GPU environment
CUDA 10.1
## create a new conda environment
conda create -n HGDrug python=3.7.3

conda activate HGDrug
## download some packages
conda install -c rdkit rdkit (contruct DSMN need)

windows:
pip install https://download.pytorch.org/whl/cu101/torch-1.4.0-cp37-cp37m-win_amd64.whl
pip install https://download.pytorch.org/whl/cu101/torchvision-0.5.0-cp37-cp37m-win_amd64.whl

linux:
pip install https://download.pytorch.org/whl/cu101/torch-1.4.0-cp37-cp37m-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu101/torchvision-0.5.0-cp37-cp37m-linux_x86_64.whl

pip install -r requirements.txt
source activate HGDrug
# Data Sets
`./DDI_data`
`./DTI_data`
`./DDiI_data`
`./DSI_data`

# Run model 
`python main.py model.conf`

# Change prediction tasks
 If the users need change the prediction task, the instructions in the model.conf need to be modified.
 Drug-drug interactions, the instructions need to be modified is as follows:
 `DFI=./DDI_data/DFI.txt`
 `FFI=./DDI_data/FFI.txt`
 `Task=./DDI_data/DDiI.txt`
 `Task.name=DrugDrug`
 
 Drug-target interactions, the instructions need to be modified is as follows:
 `DFI=./DTI_data/DFI.txt`
 `FFI=./DTI_data/FFI.txt`
 `Task=./DTI_data/DDiI.txt`
 `Task.name=DrugTarget`
 
Drug-disease interactions, the instructions need to be modified is as follows:
 `DFI=./DDiI_data/DFI.txt`
 `FFI=./DDiI_data/FFI.txt`
 `Task=./DDiI_data/DDiI.txt`
 `Task.name=DrugDisese`

 Drug-sideeffect interactions, the instructions need to be modified is as follows:
 `DFI=./DDiI_data/DFI.txt`
 `FFI=./DDiI_data/FFI.txt`
 `Task=./DDiI_data/DDiI.txt`
 `Task.name=DrugDisese`

 Drug-disease interactions, the instructions need to be modified is as follows:
 `DFI=./DSI_data/DFI.txt`
 `FFI=./DSI_data/FFI.txt`
 `Task=./DSI_data/DDiI.txt`
 `Task.name=DrugSideeffect`

# Output
Output as the results of "AUROC" and "AUPR".

