### Please check the requirements.txt and install the following packages.

nltk - `pip3 install --user nltk`

tensorflow - `pip3 install --user tensorflow`

keras - `pip3 install --user keras`

keras_contrib - `pip3 install --user git+https://www.github.com/keras-team/keras-contrib.git`


### Instructions

1. Create `splits`, `models` and `output` folders if not present. 
2. Execute `sh train_mallet.sh` first to create the models.
3. Execute `get_scores.sh` to get the test scores.
4. Execute `python3 neural.py` to run the Bi-LSTM with CRF model.
