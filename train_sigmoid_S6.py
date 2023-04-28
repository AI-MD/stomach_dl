import torch


import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from cbam_resnet.cbm_resnet import ResidualNet
from torchvision.models import resnet34, densenet121
import time, copy, argparse
from torchsummary import summary
from Dataset.StomachDataset_onehotEncoding import StomachDataset
from torchvision import transforms
from autoaugment import ImageNetPolicy,CIFAR10Policy

from senet.se_resnet import se_resnet18, se_resnet34, se_resnet50
from efficientnet_pytorch.models import EfficientNet
from sklearn.metrics import f1_score
from torchvision.models import resnet18, resnet34, resnet50, densenet121, shufflenet_v2_x0_5
from util.utils import acc
import numpy as np
# Construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=False, help="Training mode: finetune/scratch", default='scratch')
ap.add_argument("--model", required=False,
                help="model: resnet18/resnet34/resnet50/densenet121/se_resnet18/se_resnet34/se_resnet50/cbm_resnet34/cbm_resnet50/efficientnet-b0/efficientnet-b1",
                default='se_resnet50')
ap.add_argument("--train_dir", required=False, help="train dir", default='./stomach_mid_new_S6_1223/train')
ap.add_argument("--test_dir", required=False, help="test dir", default='./stomach_mid_new_S6_1223/test')
ap.add_argument("--model_save_path", required=True, help="model_save_path")
ap.add_argument("--train_bs", required=False, help="train_bs", default='16', type=int)
ap.add_argument("--valid_bs", required=False, help="valid_bs", default='8', type=int)
ap.add_argument("--epoch", required=False, help="epoch", default='50', type=int)
ap.add_argument("--optimizer", required=False, help="optimizer : SGD/Adam/AdamW", default='AdamW')
ap.add_argument("--scheduler", required=False, help="scheduler : StepLR/CosineAnnealingLR/CosineAnnealingWarmRestarts",
                default='CosineAnnealingWarmRestarts')
ap.add_argument("--input_size", required=False, help="input_size", default='224', type=int)

args = vars(ap.parse_args())

# Set training mode
train_mode = args["mode"]
model = args["model"]
# Set the train and validation directory paths
train_directory = args["train_dir"]
valid_directory = args["test_dir"]
# Set the model save path

PATH = args["model_save_path"]

# Batch size
bs = args["train_bs"]
valid_bs = args["valid_bs"]
# Number of epochs
num_epochs = args["epoch"]
# Number of classes

inpu_size=(args["input_size"], args["input_size"])

transforms_train = transforms.Compose([transforms.Resize(inpu_size),
                                       CIFAR10Policy(),
                                       # transforms.RandomHorizontalFlip(),
                                       # transforms.RandomVerticalFlip(),
                                       # transforms.ColorJitter(contrast=(0.2, 3)),
                                       transforms.ToTensor(),
                                       #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                       ]
                                      )

transforms_valid = transforms.Compose([transforms.Resize(inpu_size),
                                       transforms.ToTensor(),
                                       #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                       ]
                                      )

# Load data from folders
dataset = {
    'train': StomachDataset(data_set_path=train_directory, transforms=transforms_train),
    'valid': StomachDataset(data_set_path=valid_directory, transforms=transforms_valid)
}

# Size of train and validation data
dataset_sizes = {
    'train': len(dataset['train']),
    'valid': len(dataset['valid'])
}
#



# # Create iterators for data loading
dataloaders = {
    'train': data.DataLoader(dataset['train'], batch_size=bs, shuffle=True,
                             num_workers=0, pin_memory=True, drop_last=True),
    'valid': data.DataLoader(dataset['valid'], batch_size=valid_bs, shuffle=True,
                             num_workers=0, pin_memory=True, drop_last=True)
}

# Class names or target labels
class_names = dataset['train'].classes
print("Classes:", class_names)

num_classes = len(class_names)

# Print the train and validation data sizes
print("Training-set size:", dataset_sizes['train'],
      "\nValidation-set size:", dataset_sizes['valid'])

# Set default device as gpu, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


if train_mode == 'finetune':
    if model == "se_resnet50":
        model_ft = se_resnet50(num_classes, pretrained=True)
    elif model == "efficientnet-b0":
        model_ft = EfficientNet.from_pretrained('efficientnet-b0')
        model_ft._fc = nn.Linear(model_ft._fc.in_features, num_classes)
    elif model == "efficientnet-b1":
        model_ft = EfficientNet.from_pretrained('efficientnet-b1')
        model_ft._fc = nn.Linear(model_ft._fc.in_features, num_classes)
    elif model == "resnet18":
        model_ft = resnet18(pretrained=True)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes)
    elif model == "resnet34":
        model_ft = resnet34(pretrained=True)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes)
    elif model == "resnet50":
        model_ft = resnet50(pretrained=True)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes)
    elif model == "densenet121":
        model_ft = densenet121(pretrained=True)
        model_ft.classifier = nn.Linear(model_ft.classifier.in_features, num_classes)
    elif model=="cbm_resnet50":
        model_ft = ResidualNet('ImageNet', 50, num_classes, "CBAM")



elif train_mode == 'scratch':
    if model == "se_resnet18":
        model_ft = se_resnet18(num_classes)
    elif model == "se_resnet34":
        model_ft = se_resnet34(num_classes)
    elif model == "se_resnet50":
        model_ft = se_resnet50(num_classes, pretrained=False)
    elif model == "cbm_resnet34":
        model_ft = ResidualNet("ImageNet", 34, num_classes, 'CBAM')
    elif model == "efficientnet-b0":
        model_ft = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
    elif model == "efficientnet-b1":
        model_ft = EfficientNet.from_name('efficientnet-b1', num_classes=num_classes)
    elif model == "resnet18":
        model_ft = resnet18()
        model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes)
    elif model == "resnet34":
        model_ft = resnet34()
        model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes)
    elif model == "resnet50":
        model_ft = resnet50()
        model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes)
    elif model == "densenet121":
        model_ft = densenet121()
        model_ft.classifier = nn.Linear(model_ft.classifier.in_features, num_classes)
    elif model == "shufflenet_v2_x0_5":
        model_ft = shufflenet_v2_x0_5()
        model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes)
    elif model=="cbm_resnet50":
        model_ft = ResidualNet('ImageNet', 50, num_classes, "CBAM")
# Transfer the model to GPU
model_ft = model_ft.to(device)

# # Print model summary
# print('Model Summary:-\n')
# for num, (name, param) in enumerate(model_ft.named_parameters()):
#     print(num, name, param.requires_grad)
# summary(model_ft, input_size=(3, 224, 224))
# print(model_ft)

# Loss function
criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([40/dataset_sizes['train'],317/dataset_sizes['train'],528/dataset_sizes['train'],
                                         269 / dataset_sizes['train'], 351 / dataset_sizes['train'], 360 / dataset_sizes['train']
                                         ,299 / dataset_sizes['train'],412 / dataset_sizes['train'], 52 / dataset_sizes['train']]).to(device))

optimizer = args["optimizer"]
if optimizer == "SGD":
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, weight_decay=0.005, momentum=0.9)
elif optimizer == "Adam":
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, weight_decay=1e-5)
elif optimizer == "AdamW":
    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.001, weight_decay=1e-5)

scheduler = args["scheduler"]
if scheduler == "CosineAnnealingLR":
    scheduler_fit = lr_scheduler.CosineAnnealingLR(optimizer_ft, dataset_sizes['train'], eta_min=0.001)
elif scheduler == "CosineAnnealingWarmRestarts":
    scheduler_fit = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ft, T_0=max(2, num_epochs // 4), eta_min=1e-6)
elif scheduler == "StepLR":
    scheduler_fit = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# center loss

# Model training routine
print("\nTraining:-\n")

def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Tensorboard summary
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
                data_size = dataset_sizes[phase] // bs
                all_data_size = data_size * bs
            else:
                model.eval()  # Set model to evaluate mode
                data_size = dataset_sizes[phase] // valid_bs
                all_data_size = data_size * valid_bs

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                N, C = labels.shape
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # feature
                    features, outputs = model(inputs)

                    #_, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    outputs = torch.sigmoid(outputs)  # <--- since you use BCEWithLogitsLoss
                    # round up and down to either 1 or 0
                    predicted = torch.round(outputs)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics  # calculate how many images were correctly classified
                running_loss += loss.item() * inputs.size(0)
                running_corrects += f1_score(labels.cpu().to(torch.int).numpy(), predicted.cpu().to(torch.int).numpy(),average="samples") * inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / all_data_size

            if running_corrects == 0:
                epoch_acc = 0.00
            else:
                epoch_acc = running_corrects / (all_data_size)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Record training loss and accuracy for each phase
            if phase == 'train':
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
                writer.flush()
            else:
                writer.add_scalar('Valid/Loss', epoch_loss, epoch)
                writer.add_scalar('Valid/Accuracy', epoch_acc, epoch)
                writer.flush()

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Train the model
model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler_fit, num_epochs=num_epochs)
# Save the entire model
print("\nSaving the model...")
torch.save(model_ft, PATH)


