## Demo

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
