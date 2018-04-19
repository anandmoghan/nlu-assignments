#!/usr/bin/env bash

python3 make_data.py
echo "*************Simple Word Tagger****************"
java -cp "./mallet/class:./mallet/lib/mallet-deps.jar" cc.mallet.fst.SimpleTagger --train true --threads 6 --model-file models/simple_model.mlt splits/simple_train.txt

echo "*************Embedding Tagger****************"
java -cp "./mallet/class:./mallet/lib/mallet-deps.jar" cc.mallet.fst.SimpleTagger --train true --threads 6 --model-file models/emb_model.mlt splits/emb_train.txt

echo "*************Simple Word and Embedding Tagger****************"
java -cp "./mallet/class:./mallet/lib/mallet-deps.jar" cc.mallet.fst.SimpleTagger --train true --threads 6 --model-file models/simple_emb_model.mlt splits/simple_emb_train.txt

echo "*************Simple Word and POS Tagger****************"
java -cp "./mallet/class:./mallet/lib/mallet-deps.jar" cc.mallet.fst.SimpleTagger --train true --threads 6 --model-file models/simple_pos_model.mlt splits/simple_pos_train.txt