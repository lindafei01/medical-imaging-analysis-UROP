```
opt.federated_setting={
    'num_users':4,'frac':0.5,'local_bs':1,'iid':0,'unequal':1,'local_epoch':5, 'fed_batch_size':1, 'seperate_test_data':1
}
```

| dataset | Lower-bound Local | FedAvg | Upper-bound Data-centralized |
| :------: | :------: | :------: | :------: |
| ADNI_DX 1|0.868  |0.895  |  0.851 |
| ADNI_DX 2| 0.724 | 0.789 | 0.851  |
| ADNI_DX 3| 0.816 |  0.789 |  0.851 |
| ADNI_DX 4| 1.000 | 0.895  |  0.851 |
| COBRE 1 | 1.000 | 1.000 | 0.531  |
| COBRE 2 | 0.000 | 0.000  |  0.531 |
| COBRE 3 | 0.875 | 0.875 |  0.531 |
| COBRE 4 | 0.000 | 0.000 | 0.531  |
| ATLAS2 1 | 0.565  |  0.565    |  0.570  |
| ATLAS2 2 | 0.739 |  0.728      |  0.570 |
| ATLAS2 3 | 0.500 |     0.522   | 0.570  |
| ATLAS2 4 |  0.000|     0.000   |  0.570 |
|AIBL 1| 0.000  | 0.000 | 0.933 |
|AIBL 2| 0.077  | 0.077 | 0.933|
|AIBL 3| 0.500 | 0.500 | 0.933|
|AIBL 4|  0.000 | 0.000 | 0.933|
|ABIDE 1| 1.000 | 1.000 | 0.520  |
|ABIDE 2| 0.000 | 0.000 |  0.520 |
|ABIDE 3| 0.569 | 0.569 |  0.520 |
|ABIDE 4| 0.000 | 0.000 | 0.520  |
|MINDS 1|  0.500  |  0.500   |   0.613 |
|MINDS 2|  0.833  |   0.833  |  0.613  |
|MINDS 3|  0.000  |  0.000   |   0.613 |
|MINDS 4|  0.000  |   0.000  | 0.613   |



```text
python main.py --cuda_index 1 --method ViT --n_epoch 10 --trtype single --dataset ADNI_DX --clfsetting CN-AD --pre_datasets ADNI_DX AIBL

python main.py --cuda_index 1 --method ViT --n_epoch 10 --trtype single --dataset ADNI_DX --clfsetting CN-AD --pre_datasets ADNI_DX AIBL --method_para "{'depth': 6}"

# generate landmarks for LDMIL or DAMIDL
python main.py --no_cuda --gen_lmk --dataset ADNI_DX --clfsetting CN-AD

# RUN for 5 times and calculate the average metrics
python main.py --cuda_index 1 --method LDMIL --n_epoch 10 --trtype 5-rep --dataset ADNI_DX --clfsetting CN-AD --pre_datasets ADNI_DX AIBL

# federated
--cuda_index 1 --method ViT --n_epoch 10 --trtype single --dataset ADNI_DX --clfsetting CN-AD --pre_datasets ADNI_DX AIBL --method_para "{'depth': 6}" --federated 

#ADNI_DX
--cuda_index 3 --method ViT --global_epoch 2 --trtype single --dataset ADNI_DX --clfsetting CN-AD --pre_datasets ADNI_DX AIBL
#BraTS
--cuda_index 3 --method ViT --global_epoch 2 --trtype single --dataset BraTS --clfsetting TUMOR-CONTROL
#NIFD
--cuda_index 3 --method ViT --global_epoch 2 --trtype single --dataset NIFD --clfsetting FTD-NC
#MINDS
--cuda_index 3 --method ViT --global_epoch 2 --trtype single --dataset MINDS --clfsetting DIS-CONTROL
#ATLAS_2
--cuda_index 3 --method ViT --global_epoch 2 --trtype single --dataset ATLAS_2 --clfsetting STROKE-CONTROL
#AIBL
--cuda_index 3 --method ViT --global_epoch 2 --trtype single --dataset AIBL --clfsetting CN-AD
#PreActivationResNet18
#WideResNet18
#densenet121
```
#TODO
quality control
展示quality control以及federated learning的结果
BraTS NIFD
Visualization
multi-modal NLP:先找数据，然后处理

SiloBN FedDis
Differential Privacy, Secure Aggregation
HCI interface



联邦学习经常出现形如'AUC': nan, 'AUPR': nan, 'ACC': 1.0, 'SEN': nan, 'SPE': 1.0的情况