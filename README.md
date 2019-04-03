# robust_mtnt
Code for the NAACL 2019 paper "Improving Robustness of Machine Translation with Synthetic Noise"

#### Dependencies
```
pytorch (tested on version 0.3.0.post4)
docopt
tqdm
```

### Steps for preparing EP and MTNT data used in our experiments:

1. Fetch the data
```
mkdir data
mkdir work_dir
cd data/
wget http://www.statmt.org/europarl/v7/fr-en.tgz
wget http://www.statmt.org/wmt15/dev-v2.tgz
wget http://www.cs.cmu.edu/~pmichel1/hosting/MTNT.1.0.tar.gz
```

The training data consists of fr-en parallel data from europarl. The noisy data is collected from Reddit (https://arxiv.org/abs/1809.00388).

2. Untar the files
```
tar -xvzf fr-en.tgz
tar -xvzf dev-v2.tgz
tar xvzf MTNT.1.0.tar.gz
```

3. Preparing the data for training and testing includes selecting sentences of length less than 50, selecting subset of data and removing html tags
```
cd ../
sed '/<seg/!d' data/dev/newsdiscussdev2015-fren-ref.en.sgm | sed -e 's/\s*<[^>]*>\s*//g' > data/dev/dev.en
sed '/<seg/!d' data/dev/newsdiscussdev2015-fren-src.fr.sgm | sed -e 's/\s*<[^>]*>\s*//g' > data/dev/dev.fr
python prune_sentences.py data/europarl-v7.fr-en.fr data/europarl-v7-50.fr-en.fr data/europarl-v7.fr-en.en data/europarl-v7-50.fr-en.en 50
head -n1000000 data/europarl-v7-50.fr-en.fr > train.fr
head -n1000000 data/europarl-v7-50.fr-en.en > train.en
cut -f2 -d$'\t' MTNT/train/train.fr-en.tsv > train.mtnt.fr
cut -f3 -d$'\t' MTNT/train/train.fr-en.tsv > train.mtnt.en
cut -f2 -d$'\t' MTNT/valid/valid.fr-en.tsv > dev.mtnt.fr
cut -f3 -d$'\t' MTNT/valid/valid.fr-en.tsv > dev.mtnt.en
cut -f3 -d$'\t' MTNT/test/test.fr-en.tsv > test.mtnt.en
cut -f2 -d$'\t' MTNT/test/test.fr-en.tsv > test.mtnt.fr
```

4. Use spe encoding to create sub-word data. Highly recommended ! Below command assumes you have spe models placed in _sp_models/_, specifically, _europarl-v7.fr-en.en.model_ and _europarl-v7.fr-en.fr.model_. We have provided these files in the repository with vocab size 16k.
```
python encode_spm.py -m sp_models/europarl-v7.fr-en.fr.model -i data/train.fr -o data/train.tok.fr
python encode_spm.py -m sp_models/europarl-v7.fr-en.en.model -i data/train.en -o data/train.tok.en
python encode_spm.py -m sp_models/europarl-v7.fr-en.fr.model -i data/dev/dev.fr -o data/dev/dev.tok.fr
python encode_spm.py -m sp_models/europarl-v7.fr-en.en.model -i data/dev/dev.en -o data/dev/dev.tok.en

python encode_spm.py -m sp_models/europarl-v7.fr-en.fr.model -i data/test.mtnt.fr -o data/test.mtnt.tok.fr
```

### Steps for training the baseline model:
 
1. Training the model.
```
python vocab.py --train-src=data/train.tok.fr --train-tgt=data/train.tok.en data/vocab-bpe.bin --freq-cutoff 0
python nmt.py train --train-src="data/train.tok.fr" --train-tgt="data/train.tok.en" --dev-src="data/dev/dev.tok.fr" --dev-tgt="data/dev/dev.tok.en" --vocab="data/vocab.bin" --save-to="work_dir/" --valid-niter=1000 --batch-size=32 --hidden-size=256 --embed-size=512  --optim=1 --max-epoch=30 --uniform-init=0.1 --dropout=0.3 --lr=0.01 --clip-grad=20 --lr-decay=0.5 --patience=3 --tie-weights=1 --n_layers=2
```

2. Decoding on the dev set, once the model is trained. Replace the model file name with the correct name.
```
python nmt.py decode --beam-size=5 --max-decoding-time-step=100 --embed-size=512 --tie-weights=1 --n_layers=2 --vocab="data/vocab-bpe.bin" "work_dir/model_epoch.t7" "data/dev/dev/dev.tok.fr" "work_dir/decode-fr-en.tok.txt"
python decode_spm.py -m sp_models/europarl-v7.fr-en.en.model -i work_dir/decode-fr-en.tok.txt -o work_dir/decode-fr-en.txt
```

3. Compute the bleu score using the decoded file _decode-fr-en.txt_.
```
perl multi-bleu.perl "data/dev/dev.en" < "work_dir/decode-fr-en.txt"
```

### Steps for generating _EP-100k-SNI_:

1. Randomly sample 100k parallel examples from _train.en_ and _train.fr_ to create _train-100k.en_ and _train-100k.fr_. 
An easy way to do this would be to just pick the top 100k examples (this might not be most representative of the entire dataset).

```
head -n100000 data/train.fr > data/train-100k.fr
head -n100000 data/train.en > data/train-100k.en
```

2. Add random noise in this smaller dataset. We used the same smaller dataset across all the methods proposed in the paper.

```
python artificial_noise.py data/train-100k.fr data/train-100k.en data/train-100k.sni.fr data/train.sni.en "0.04,0.007,0.002,0.015"
python encode_spm.py -m sp_models/europarl-v7.fr-en.fr.model -i data/train.sni.fr -o data/train.sni.tok.fr
python encode_spm.py -m sp_models/europarl-v7.fr-en.en.model -i data/train.sni.en -o data/train.sni.tok.en
```

### Steps for generating _EP-100k-UBT_:

1. Get the TED talks data for training the two intermediate models from http://phontron.com/data/ted_talks.tar.gz
2. Use _ted_reader.py_ provided in the repository https://github.com/neulab/word-embeddings-for-nmt for extracting parallel corpora for _en-fr_.
3. Make sure the following files exist in the folder _ted_data/_, _train.fr_, _train.en_, _dev.fr_ and _dev.en_ (these files will be obtained using the above two steps)
4. Prune out sentences with length more than 50.
```
python prune_sentences.py ted_data/train.fr ted_data/train-50.fr ted_data/train.en ted_data/train-50.en 50
```
5. Encode the data using the spe model trained on ted data (spe model provided).
```
python encode_spm.py -m sp_models/ted.fr-en.en.model -i ted_data/train-50.en -o ted_data/train.tok.en
python encode_spm.py -m sp_models/ted.fr-en.fr.model -i ted_data/train-50.fr -o ted_data/train.tok.fr
python encode_spm.py -m sp_models/ted.fr-en.en.model -i ted_data/dev.en -o ted_data/dev.tok.en
python encode_spm.py -m sp_models/ted.fr-en.fr.model -i ted_data/dev.fr -o ted_data/dev.tok.fr
python encode_spm.py -m sp_models/ted.fr-en.fr.model -i data/test.mtnt.fr -o ted_data/test.mtnt.tok.fr
```
6. Concatenate the MTNT training data.
```
cat ted_data/train.tok.fr MTNT/train/train-50.mtnt.tok.fr > ted_data/train_ted_mtnt.tok.fr
cat ted_data/train.tok.en MTNT/train/train-50.mtnt.tok.en > ted_data/train_ted_mtnt.tok.en
```
6. Train the model in forward and backward direction using the concatenated data and the commands mentioned for training the baseline model.
7. To get the noisy data using UBT i.e _EP-100k-UBT_, just decode using the best models obtained in the previous step.
```
python encode_spm.py -m sp_models/ted.fr-en.fr.model -i data/train-100k.fr -o data/train-100k.ted.tok.fr
python encode_spm.py -m sp_models/ted.fr-en.en.model -i data/train-100k.en -o data/train-100k.ted.tok.en

python nmt.py decode --beam-size=5 --max-decoding-time-step=100 --embed-size=512 --tie-weights=1 --n_layers=2 --vocab="vocab-ted-fr-en.bin" "work_dir/model_ted_best_forward.t7" "data/train-100k.ted.tok.fr" "work_dir/ted.decode-fr-en.tok.en"

python nmt.py decode --beam-size=5 --max-decoding-time-step=100 --embed-size=512 --tie-weights=1 --n_layers=2 --vocab="ted_data/vocab-spe-reverse.bin" "work_dir/model_ted_best_backward.t7" "work_dir/ted.decode-fr-en.tok.en" "work_dir/ted.decode-fr-en.tok.fr"

python decode_spm.py -m sp_models/ted.fr-en.en.model -i work_dir/ted.decode-fr-en.tok.en -o work_dir/ted.train.decode-fr-en.en
python decode_spm.py -m sp_models/ted.fr-en.fr.model -i work_dir/ted.decode-fr-en.tok.fr -o work_dir/ted.train.decode-fr-en.fr
```

8. The files _ted.train.decode-fr-en.en_ and _ted.train.decode-fr-en.fr_ are used in finetuning for the UBT method.

### Steps for generating _EP-100k-TBT_:

1. Follow the first five steps of UBT to get the required data.
2. Prior to concatenating the data, append the data source tag.
```
sed 's/^/_TED_ /g' ted_data/train.tok.fr > ted_data/train.tag.tok.fr
sed 's/^/_MTNT_ /g' MTNT/train/train-50.mtnt.tok.fr > MTNT/train/train-50.mtnt.tag.tok.fr
```
3. Concatenate the above two files for _fr_ and _en_ respectively.
```
cat ted_data/train.tag.tok.fr MTNT/train/train-50.mtnt.tag.tok.fr > ted_data/train_ted_mtnt.tag.tok.fr
```
4. Use the files obtained in the above step to train the forward and backward model.
5. To get the noisy data using TBT i.e _EP-100k-TBT_, just decode using the best models obtained in the previous step. But append the noisy data source tag before decoding.
```
sed 's/^/_MTNT_ /g' data/train-100k.ted.tok.fr > data/train-100k.ted.tag.tok.fr
sed 's/^/_MTNT_ /g' data/train-100k.ted.tok.en > data/train-100k.ted.tag.tok.en

python nmt.py decode --beam-size=5 --max-decoding-time-step=100 --embed-size=512 --tie-weights=1 --n_layers=2 --vocab="vocab-ted-fr-en.bin" "work_dir/model_ted_best_forward.t7" "data/train-100k.ted.tag.tok.fr" "work_dir/ted.decode-fr-en.tag.tok.en"

python nmt.py decode --beam-size=5 --max-decoding-time-step=100 --embed-size=512 --tie-weights=1 --n_layers=2 --vocab="ted_data/vocab-spe-reverse.bin" "work_dir/model_ted_best_backward.t7" "work_dir/ted.decode-fr-en.tag.tok.en" "work_dir/ted.decode-fr-en.tag.tok.fr"

python decode_spm.py -m sp_models/ted.fr-en.en.model -i work_dir/ted.decode-fr-en.tag.tok.en -o work_dir/ted.train.decode-fr-en.en
python decode_spm.py -m sp_models/ted.fr-en.fr.model -i work_dir/ted.decode-fr-en.tag.tok.fr -o work_dir/ted.train.decode-fr-en.fr
```
6. The files _ted.train.decode-fr-en.en_ and _ted.train.decode-fr-en.fr_ are used in finetuning for the TBT method.

### Steps for fine-tuning an existing model with noisy data:

It is same as training a new model using the additional data but with the weights loaded from a pre-trained model.
```
python nmt.py train --train-src="data/train.sni.tok.fr" --train-tgt="data/train.sni.tok.en" --dev-src="data/dev/dev.tok.fr" --dev-tgt="data/dev/dev.tok.en" --vocab="data/vocab.bin" --save-to="work_dir/" --valid-niter=1000 --batch-size=32 --hidden-size=256 --embed-size=512  --optim=1 --max-epoch=30 --uniform-init=0.1 --dropout=0.3 --lr=0.01 --clip-grad=20 --lr-decay=0.5 --patience=3 --tie-weights=1 --n_layers=2 --load-weights-from "work_dir/model_baseline.t7"
```

If you use the code, please consider citing the paper using following bibtex:

#### BibTex
```
@inproceedings{vaibhav19naacl,
    title = {Improving Robustness of Machine Translation with Synthetic Noise},
    author = {Vaibhav and Sumeet Singh and Craig Stewart and Graham Neubig},
    booktitle = {Meeting of the North American Chapter of the Association for Computational Linguistics (NAACL)},
    address = {Minneapolis, USA},
    month = {June},
    year = {2019}
}
```
