dataset=MAG
metagraph=PRP

green=`tput setaf 2`
reset=`tput sgr0`

echo "${green}=====Step 1: Preparing Testing Data=====${reset}"
python3 prepare_test.py --dataset ${dataset}

echo "${green}=====Step 2: Generating Training Data=====${reset}"
python3 prepare_train.py --dataset ${dataset} --metagraph ${metagraph}

head -100000 ${dataset}_input/dataset.txt > ${dataset}_input/train.txt
sed -n '100001,110000p' ${dataset}_input/dataset.txt > ${dataset}_input/dev.txt
