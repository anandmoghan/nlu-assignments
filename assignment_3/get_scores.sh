#!/usr/bin/env bash

echo "*************Simple Word Tagger****************"
java -cp "./mallet/class:./mallet/lib/mallet-deps.jar" cc.mallet.fst.SimpleTagger --train false --threads 6 --model-file models/simple_model.mlt splits/simple_test.txt > output/simple_pred.txt
python3 test_scores.py --true_test splits/true_test.txt --pred_test output/simple_pred.txt

echo "*************Embedding Tagger****************"
java -cp "./mallet/class:./mallet/lib/mallet-deps.jar" cc.mallet.fst.SimpleTagger --train false --threads 6 --model-file models/emb_model.mlt splits/emb_test.txt > output/emb_pred.txt
python3 test_scores.py --true_test splits/true_test.txt --pred_test output/emb_pred.txt

echo "*************Simple Word and Embedding Tagger****************"
java -cp "./mallet/class:./mallet/lib/mallet-deps.jar" cc.mallet.fst.SimpleTagger --train false --threads 6 --model-file models/simple_emb_model.mlt splits/simple_emb_test.txt > output/simple_emb_pred.txt
python3 test_scores.py --true_test splits/true_test.txt --pred_test output/simple_emb_pred.txt

echo "*************Simple Word and POS Tagger****************"
java -cp "./mallet/class:./mallet/lib/mallet-deps.jar" cc.mallet.fst.SimpleTagger --train false --threads 6 --model-file models/simple_pos_model.mlt splits/simple_pos_test.txt > output/simple_pos_pred.txt
python3 test_scores.py --true_test splits/true_test.txt --pred_test output/simple_pos_pred.txt