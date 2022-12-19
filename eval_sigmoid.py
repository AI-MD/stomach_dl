import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import multiprocessing
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score , precision_score,recall_score,roc_auc_score

import sys
from Dataset.StomachDataset_onehotEncoding import StomachDataset

# Paths for image directory and model
EVAL_DIR = sys.argv[1]
EVAL_MODEL= sys.argv[2]
input_size = int(sys.argv[3])
# Load the model for evaluation
model = torch.load(EVAL_MODEL)
model.eval()

# Configure batch size and nuber of cpu's

bs = 1
inpu_size=(input_size, input_size)
# Prepare the eval data loader
eval_transform=transforms.Compose([transforms.Resize(inpu_size),
                                   transforms.ToTensor()

                                  ])

eval_dataset=StomachDataset(data_set_path=EVAL_DIR, transforms=eval_transform)
eval_loader=data.DataLoader(eval_dataset, batch_size=bs, shuffle=False,
                            num_workers=0, pin_memory=True)

# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Number of classes and dataset-size
num_classes=len(eval_dataset.classes)
dsize=len(eval_dataset)

# Class label names
# class_names=['D1', 'D2', 'D3', 'E1', 'E2', 'E3', 'S1', 'S2', 'S3', 'S4', 'S5']
# class_names=['S1', 'S2', 'S3', 'S4', 'S5']
class_names=['C','D1','D2','E','S1', 'S2', 'S3', 'S4', 'S5', 'S6']

# Initialize the prediction and label lists
predlist=torch.zeros(0,dtype=torch.long, device='cpu')
lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

# Evaluate the model accuracy on the dataset
f1_correct = 0
precision_collect = 0
recall_collect = 0
total = 0
with torch.no_grad():
    for images, labels in eval_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        outputs = torch.sigmoid(outputs)  # <--- since you use BCEWithLogitsLoss

        #print("확률", np.round(outputs.cpu()[0], 3) , "정답", labels)

        # round up and down to either 1 or 0
        predicted = torch.round(outputs)

        # idx = np.where(outputs[0] > 0.99)[0]
        # if len(idx) > 0:
        #     for id, val in enumerate(idx):
        #         # if val == 4:
        #         #     if predicted[0][val] > 0.995:
        #         #         predicted_classes.append(class_names[val])
        #         #     else:
        #         #         predicted_classes.append("x")
        #         # else:
        #         predicted_classes.append(class_names[val])

        predlist = torch.cat([predlist, predicted.cpu()])
        lbllist = torch.cat([lbllist, labels.cpu()])

print(lbllist.numpy().shape)
print(predlist.numpy().shape)

print('roc_auc: {:.4f} '.format(roc_auc_score(lbllist.numpy(), predlist.numpy(), average='micro')))

print(' recall_score: {:.4f}'.format(recall_score(lbllist.numpy(), predlist.numpy(), average="micro")))
print(' precesion_score: {:.4f}'.format(precision_score(lbllist.numpy(), predlist.numpy(), average="micro")))
print('f1score: {:.4f} '.format(f1_score(lbllist.numpy(), predlist.numpy(), average='micro')))

conf_mat = multilabel_confusion_matrix(lbllist.numpy(), predlist.numpy())

print('multilabel Confusion Matrix')
print('-' * 16)
print(conf_mat)