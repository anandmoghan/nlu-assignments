from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import numpy as np

parser = ArgumentParser()
parser.add_argument('--true_test', help='True Test Data File')
parser.add_argument('--pred_test', help='Predicted Test Data File')
args = parser.parse_args()

test_tags = np.loadtxt(args.true_test, dtype='S')
test_pred = np.loadtxt(args.pred_test, dtype='S')

labels = sorted(list(set(test_pred)))

print('Accuracy Score', accuracy_score(test_tags, test_pred))
print('Macro F1 Score', f1_score(test_tags, test_pred, average='macro'))
print('Micro F1 Score', f1_score(test_tags, test_pred, average='micro'))
print('Weighted F1 Score', f1_score(test_tags, test_pred, average='weighted'))
print('Precision Score', precision_score(test_tags, test_pred, average='weighted'))
print('Recall Score', recall_score(test_tags, test_pred, average='weighted'))
print('Confusion Matrix\n', ' '.join(str(l) for l in labels), '\n', confusion_matrix(test_tags, test_pred, labels=labels))
