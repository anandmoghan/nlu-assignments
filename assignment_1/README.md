## Assignment 1 - Language Models

### Generate Models
Execute `python3 main.py` to generate all the models and check the perplexity in all settings. This will create `brown.json` `gutenberg.json` and `brown_gutenbergy.json` which contains all the n-grams counts which can be used for speeding up testing when you run the code again.

On execution it will ask you whether you want the model retrained on both of `brown` and `gutenberg` models if the jsons are already present. You can give (Y/N) accordingly.

### Generate Sentences
Execute `generate_sentence.sh` script to build a sentence out of the model. If the is not trained model data present, it will automatically execute `main.py` to build the model and then generates the sentence. 
