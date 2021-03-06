#the split to train
SPLIT=(1)
SHOT=(1 2 3 5 10)
Test_weights=(01 02 03 04 05 06 07 08 09 10)
##base training
mkdir backup 
for split in ${SPLIT[*]}
do
    python tool/train_decoupling_disturbance.py cfg/fewshot/metayolo_split${split}.data cfg/darknet_dynamic.cfg cfg/reweighting_net_decoupling.cfg darknet19_448.conv.23 backup/split${split}_base
done
## finetune
for shot in ${SHOT[*]}
do
  for split in ${SPLIT[*]}
  do
     python tool/train_decoupling_disturbance.py cfg/fewshot/metatune_${shot}shot_split${split}.data cfg/darknet_dynamic.cfg cfg/reweighting_net_decoupling.cfg backup/split${split}_base/000350.weights backup/split${split}_${shot}shot
  done
done
## test weight
for shot in ${SHOT[*]}
do
  for split in ${SPLIT[*]}
  do
    for test_weights in ${Test_weights[*]}
    do
       python tool/valid_decoupling.py cfg/fewshot/metatune_${shot}shot_split${split}.data cfg/darknet_dynamic.cfg cfg/reweighting_net_decoupling.cfg backup/split${split}_${shot}shot/0000${test_weights}.weights
    done
  done
done
