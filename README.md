# robust_mtnt
Code for the paper "Improving Robustness of Machine Translation with Synthetic Noise"


### Steps for preparing the _EP-100k_ data used in our experiments:

1. Fetch the data
```
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
cut -f3 -d$'\t' MTNT/test/test.fr-en.tsv > test.ntmt.en
cut -f2 -d$'\t' MTNT/test/test.fr-en.tsv > test.ntmt.fr
```

4. Use spe encoding to create sub-word data. Highly recommended ! Below command assumes you have spe models placed in _sp_models/_, specifically, _europarl-v7.fr-en.en.model_ and _europarl-v7.fr-en.fr.model_. We have provided these files in the repository with vocab size 16k.
```
python encode_spm.py -m sp_models/europarl-v7.fr-en.fr.model -i data/train.fr -o data/train.tok.fr
python encode_spm.py -m sp_models/europarl-v7.fr-en.en.model -i data/train.en -o data/train.tok.en
python encode_spm.py -m sp_models/europarl-v7.fr-en.fr.model -i data/dev/dev.fr -o data/dev/dev.tok.fr
python encode_spm.py -m sp_models/europarl-v7.fr-en.en.model -i data/dev/dev.en -o data/dev/dev.tok.en

python encode_spm.py -m sp_models/europarl-v7.fr-en.fr.model -i data/test.ntmt.fr -o data/test.ntmt.tok.fr
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

```
python artificial_noise.py data/train.fr data/train.en data/train.sni.fr data/train.sni.en "0.04,0.007,0.002,0.015"
python encode_spm.py -m sp_models/europarl-v7.fr-en.fr.model -i data/train.sni.fr -o data/train.sni.tok.fr
python encode_spm.py -m sp_models/europarl-v7.fr-en.en.model -i data/train.sni.en -o data/train.sni.tok.en
```
