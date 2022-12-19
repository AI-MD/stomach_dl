import torch

from torchvision import transforms
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from cbam_resnet.cbm_resnet import ResidualNet
from torchvision.models import resnet34, densenet121
import time, copy, argparse

from senet.se_resnet import se_resnet18,se_resnet34,se_resnet50
from torchsummary import summary
from Dataset.StomachDataset import StomachDataset

from efficientnet_pytorch.models import EfficientNet
from center_loss import CenterLoss

# Construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=False, help="Training mode: finetue/scratch", default='scratch')
ap.add_argument("--model", required=False, help="model: resnet34/densenet121/se_resnet18/se_resnet34/se_resnet50/cbm_resnet34/efficientnet-b0", default='resnet34')
ap.add_argument("--train_dir", required=False, help="train dir", default='./stomach_mid/train')
ap.add_argument("--test_dir", required=False, help="test dir", default='./stomach_mid/test')
ap.add_argument("--model_save_path", required=True, help="model_save_path")
ap.add_argument("--train_bs", required=False, help="train_bs", default='32',type=int)
ap.add_argument("--valid_bs", required=False, help="valid_bs", default='16',type=int)
ap.add_argument("--epoch", required=False, help="epoch", default='100',type=int)
ap.add_argument("--optimizer", required=False, help="optimizer : SGD/Adam/AdamW", default='AdamW')
ap.add_argument("--scheduler", required=False, help="scheduler : StepLR/CosineAnnealingLR/CosineAnnealingWarmRestarts",default='CosineAnnealingWarmRestarts')
args= vars(ap.parse_args())

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


transforms_train = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.RandomRotation(10.),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                       ]
                                      )

transforms_valid = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

if train_mode=='finetune':
    if model == "se_resnet50":
        model_ft = se_resnet50(num_classes, pretrained=True)

elif train_mode=='scratch':
    if model == "se_resnet18":
        model_ft = se_resnet18(num_classes)
    elif model == "se_resnet34":
        model_ft = se_resnet34(num_classes)
    elif model == "se_resnet50":
        model_ft = se_resnet50(num_classes,pretrained=False)
    elif model == "cbm_resnet34":
        model_ft = ResidualNet("ImageNet",34,num_classes,'CBAM')
    elif model == "efficientnet-b0":
        model_ft = EfficientNet.from_name('efficientnet-b0',num_classes=num_classes)
    elif model=="resnet34":
        model_ft = resnet34()
        model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes)
    elif model == "densenet121":
        model_ft = densenet121()
        model_ft.classifier  = nn.Linear(model_ft.fc.in_features, num_classes)

# Transfer the model to GPU
model_ft = model_ft.to(device)

# Print model summary
print('Model Summary:-\n')
for num, (name, param) in enumerate(model_ft.named_parameters()):
    print(num, name, param.requires_grad)
summary(model_ft, input_size=(3, 224, 224))
print(model_ft)

# Loss function
criterion = nn.CrossEntropyLoss()

optimizer = args["optimizer"]
if optimizer == "SGD":
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, weight_decay=0.005, momentum=0.9)
elif optimizer == "Adam":
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, weight_decay=1e-5)
elif optimizer == "AdamW":
    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.001, weight_decay=1e-5)

scheduler = args["scheduler"]
scheduler_fit = torch.cuda.amp.GradScalertorch.cuda.amp.GradScaler()


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
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                # with torch.set_grad_enabled(phase == 'train'):
                #     outputs = model(inputs)
                #     _, preds = torch.max(outputs, 1)
                #     loss = criterion(outputs, labels)
                with torch.cuda.amp.autocast():

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        scheduler.scale(loss).backward()
                        scheduler.step(optimizer)
                        scheduler.update()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            # if phase == 'train':
            #     scheduler.update()

            epoch_loss = running_loss / dataset_sizes[phase]

            if running_corrects==0:
                epoch_acc=0.00
            else:
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

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

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#Train the model
model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler_fit,num_epochs=num_epochs)
# Save the entire model
print("\nSaving the model...")
torch.save(model_ft, PATH)
