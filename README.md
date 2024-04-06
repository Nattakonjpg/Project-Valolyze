# Github main project

Created: April 6, 2024 2:01 AM
Class: AI

### 1. Connect with your Data

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Install package

```python
from sklearn.metrics import confusion_matrix
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch
import transforms as T
from movinets import MoViNet
from movinets.config import _C
```

```python
!pip install av
```

```python
pip install git+https://github.com/Atze00/MoViNet-pytorch.git
```

### 3. Identify parameter MoViNet-A5

```python
torch.manual_seed(97)
num_frames = 24 # 16
clip_steps = 12
Bs_Train = 1
Bs_Test = 1

transform_A5 = transforms.Compose([

                                 T.ToFloatTensorInZeroOne(),
                                 T.Resize((400, 400)),
                                 T.RandomHorizontalFlip(),
                                 #T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                 T.RandomCrop((320, 320))]) # Size for model
transform_test_A5 = transforms.Compose([
                                 T.ToFloatTensorInZeroOne(),
                                 T.Resize((400, 400)),
                                 #T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                 T.CenterCrop((320, 320))]) #Size for model
```

### 3.1 Train_dataset

```python
hmdb51_train_A5 = torchvision.datasets.HMDB51('video_data', 'test_train_splits', num_frames,frame_rate=None,
                                                step_between_clips = clip_steps, fold=1, train=True,
                                                transform=transform_A5, num_workers=2)
```

### 3.2 Test_dataset

```python
hmdb51_test_A5 = torchvision.datasets.HMDB51('video_data', 'test_train_splits', num_frames,frame_rate=None,
                                                step_between_clips = clip_steps, fold=1, train=False,
                                                transform=transform_test_A5, num_workers=2)
```

### 3.3 Dataloader

```python
train_loader_A5 = DataLoader(hmdb51_train_A5, batch_size=Bs_Train, shuffle=True)
test_loader_A5  = DataLoader(hmdb51_test_A5 , batch_size=Bs_Test, shuffle=False)
```

### 4 function for test model, evaluate

```python
def train_iter(model, optimz, data_load, loss_val):
    samples = len(data_load.dataset)
    model.train()
    model.cuda()
    model.clean_activation_buffers()
    optimz.zero_grad()
    for i, (data,_ , target) in enumerate(data_load):
        out = F.log_softmax(model(data.cuda()), dim=1)
        loss = F.nll_loss(out, target.cuda())
        loss.backward()
        optimz.step()
        optimz.zero_grad()
        model.clean_activation_buffers()
        if i % 50 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_val.append(loss.item())

def evaluate(model, data_load, loss_val):
    model.eval()

    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0
    model.clean_activation_buffers()
    with torch.no_grad():
        for data, _, target in data_load:
            output = F.log_softmax(model(data.cuda()), dim=1)
            loss = F.nll_loss(output, target.cuda(), reduction='sum')
            _, pred = torch.max(output, dim=1)

            tloss += loss.item()
            csamp += pred.eq(target.cuda()).sum()
            model.clean_activation_buffers()
    aloss = tloss / samples
    loss_val.append(aloss)
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')

```

### 5 train model and save model with checkpoint

```python
N_EPOCHS = 1
# สร้างโมเดล MoViNet
model_A5 = MoViNet(_C.MODEL.MoViNetA5, causal=False, pretrained=True)
start_time = time.time()

# เตรียม list เพื่อเก็บค่า loss ของการฝึกอบรม
trloss_val, tsloss_val = [], []

# ปรับแต่งชั้น classifier ของโมเดล
model_A5.classifier[3] = torch.nn.Conv3d(2048, 8, (1, 1, 1))

# กำหนด optimizer
optimizer = optim.Adam(model_A5.parameters(), lr=0.00005)

# วน loop ตามจำนวน epoch ที่กำหนด
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)

    # ทำการฝึกอบรมและบันทึกค่า loss
    loss = train_iter(model_A5, optimizer, train_loader_A5, trloss_val)

    # บันทึก checkpoint
    checkpoint_path = 'A5_habor_3_300{}.pth'.format(epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_A5.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

    # บันทึกโมเดลสุดท้าย
    if epoch == N_EPOCHS:
        torch.save(model_A5.state_dict(), 'A5_habor_3_300.pth')

print('Training Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
```

### 6. print graph loss after train model

```python
import matplotlib.pyplot as plt
N_EPOCHS = len(trloss_val)

# Plotting the training loss
plt.plot(range(1, N_EPOCHS + 1), trloss_val, label='Training Loss')

# Adding labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')

# Adding a legend
plt.legend()

# Display the plot
plt.show()
```

### 7. predict datatest

```python
device = next(model_A5.parameters()).device
true_labels, predicted_labels = [], []
with torch.no_grad():
    for data, _, target in test_loader_A5:
        # Move data and target to the same device as the model
        data, target = data.to(device), target.to(device)

        output = F.log_softmax(model_A5(data), dim=1)
        _, predicted = torch.max(output, dim=1)

        true_labels.extend(target.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())
```

### 7.1 Save test data

```python
import pandas as pd
# สร้าง DataFrame จาก true_labels และ predicted_labels
df = pd.DataFrame({'true_label': true_labels, 'predicted_label': predicted_labels})
# บันทึก DataFrame เป็นไฟล์ CSV
df.to_csv('true_predict_A5_habor_habor.csv', index=False)
```

### 8. Confusiton matrix + package

```python
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import torch
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
```

```python
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)
```

### 9. Confusion matrix graph

```python
from prettytable import PrettyTable

# Assuming binary classification (positive and negative)
positive_class_index = 1  # Index of the positive class in your class_names list

# Extract relevant values from the confusion matrix
true_positive = conf_matrix[positive_class_index, positive_class_index]
false_positive = conf_matrix[:, positive_class_index].sum() - true_positive
false_negative = conf_matrix[positive_class_index, :].sum() - true_positive
true_negative = conf_matrix.sum() - (true_positive + false_positive + false_negative)

# Create a PrettyTable
table = PrettyTable(["", "Predicted Positive", "Predicted Negative", "Total"])
table.add_row(["Actual Positive", true_positive, false_negative, true_positive + false_negative])
table.add_row(["Actual Negative", false_positive, true_negative, false_positive + true_negative])
table.add_row(["Total", true_positive + false_positive, false_negative + true_negative, conf_matrix.sum()])
# Print the table
print(table)
```

```python
  # Print classification report
  print("Classification Report:")
  print(classification_report(true_labels, predicted_labels));
```

```python
#class_names_test = ['Buy_Action', 'Class 0 Action', 'Class Skill Action', 'Dead_Action', 'Defuse Action', 'Orb Action', 'Plant Action', 'Shoot Action']
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
```

```python
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming you have true_labels and predicted_labels calculated

# Custom class labels
#class_labels =['Buy_Action', 'Class 0 Action', 'Class Skill Action', 'Dead_Action', 'Defuse Action', 'Orb Action', 'Plant Action', 'Shoot Action']
class_labels = class_names

# Calculate precision, recall, f1-score, and accuracy for each class
precision_per_class = precision_score(true_labels, predicted_labels, average=None)
recall_per_class = recall_score(true_labels, predicted_labels, average=None)
f1_per_class = f1_score(true_labels, predicted_labels, average=None)
accuracy_per_class = accuracy_score(true_labels, predicted_labels)

# Create PrettyTable
table = PrettyTable(["Class", "Precision", "Recall", "F1-Score", "Accuracy"])

# Add rows to the table for each class
for class_label, precision, recall, f1 in zip(class_labels, precision_per_class, recall_per_class, f1_per_class):
    table.add_row([class_label, f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", ""])

# Add a row for the overall accuracy
table.add_row(["Overall", "", "", "", f"{accuracy_per_class:.4f}"])
# Display the table
print(table)
```

### 10. Section test with validation data with 2 sec per video

```python
import time
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch
import transforms as T
from movinets import MoViNet
from movinets.config import _C
```

install package

### 11. need ffmpeg for cutting video

```python
from IPython.display import clear_output
!sudo curl -L https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz -o /usr/local/bin/ffmpeg.tar.xz
clear_output()
%cd /usr/local/bin/
clear_output()
!7z e /usr/local/bin/ffmpeg.tar.xz
clear_output()
!7z e /usr/local/bin/ffmpeg.tar
clear_output()
!sudo chmod a+rx /usr/local/bin/ffmpeg
clear_output()
%cd /content/
!sudo curl -L https://mkvtoolnix.download/appimage/MKVToolNix_GUI-70.0.0-x86_64.AppImage -o /usr/local/bin/MKVToolNix_GUI-70.0.0-x86_64.AppImage
!sudo chmod u+rx /usr/local/bin/MKVToolNix_GUI-70.0.0-x86_64.AppImage
!sudo ln -s /usr/local/bin/MKVToolNix_GUI-70.0.0-x86_64.AppImage /usr/local/bin/mkvmerge
!sudo chmod a+rx /usr/local/bin/mkvmerge
clear_output()
!ffmpeg -version
```

### 12. Load model before test validation with gpu in colab

```python
# Cuda , use Gpu for predict
loaded_model = MoViNet(_C.MODEL.MoViNetA5, causal=False, pretrained=True)
loaded_model.classifier[3] = torch.nn.Conv3d(2048, 8, (1, 1, 1))
# ระบุ path ของ checkpoint ที่คุณต้องการโหลด
best_checkpoint_path = 'your_path_model.pth'
# Load checkpoint
best_checkpoint = torch.load(best_checkpoint_path)
# Load model state dictionary
loaded_model.load_state_dict(best_checkpoint['model_state_dict'])
# ทดสอบโมเดล
loaded_model.eval()
```

### 12.1 if your colab is not colab pro or full of gpu use this !

```python
# Cpu
loaded_model = MoViNet(_C.MODEL.MoViNetA5, causal=False, pretrained=True)
loaded_model.classifier[3] = torch.nn.Conv3d(2048, 8, (1, 1, 1))

# Specify the path to the checkpoint file
best_checkpoint_path = 'your_path_model.pth'

# Load the checkpoint, specifying map_location to CPU
best_checkpoint = torch.load(best_checkpoint_path, map_location=torch.device('cpu'))

# Load the model's state dictionary
loaded_model.load_state_dict(best_checkpoint['model_state_dict'])

# Set the model to evaluation mode
loaded_model.eval()
```

### 13 All function for predict

1. function cut video + create text.csv ( identify time in video )

```python
def cut_video(input_file, output_folder, clip_duration, output_csv):
    # Get video duration
    command = ['ffmpeg', '-i', input_file, '-f', 'null', '-']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = result.stdout.decode('utf-8')

    # Check if output contains duration information
    if 'Duration: ' not in output:
        print("Error: Failed to get video duration.")
        return

    duration_index = output.find('Duration: ') + len('Duration: ')
    duration = output[duration_index:duration_index+11]
    duration = duration.split(':')
    duration = int(duration[0])*3600 + int(duration[1])*60 + int(duration[2].split('.')[0])

    # Calculate number of clips
    num_clips = duration // clip_duration

    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    # Create DataFrame to store clip start times
    clip_start_times = []

    # Cut each clip
    for i in range(num_clips):
        start_time = i * clip_duration
        end_time = start_time + clip_duration

        # Generate command line for FFMPEG
        command_line = ['-hide_banner', '-i', input_file, '-map', '0', '-c', 'copy', '-ss',
                        "{:02d}:{:02d}:{:02d}".format(start_time // 3600, (start_time // 60) % 60, start_time % 60),
                        '-to',
                        "{:02d}:{:02d}:{:02d}".format(end_time // 3600, (end_time // 60) % 60, end_time % 60)]

        # Set output file path
        output_file = os.path.join(output_folder, f"video_test_{i+1}.avi")
        command_line += [output_file]

        # Run FFMPEG to cut the video
        subprocess.run(['ffmpeg'] + command_line)

        # Store clip start time
        clip_start_times.append("{:02d}:{:02d}:{:02d}".format(start_time // 3600, (start_time // 60) % 60, start_time % 60))

    # Create DataFrame with clip start times
    df = pd.DataFrame({'clip_start_times': clip_start_times})
    # Save DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
```

1. function create text (below)

```python
def create_test_split(input_path, output_file_path):
    with open(output_file_path, "w") as f:
        for i, filename in enumerate(os.listdir(input_path), start=1):
            if filename.endswith(".avi"):
                new_filename = f"video_test_{i}.avi 2\n"
                f.write(new_filename)
```

1. function dataloder

```python
torch.manual_seed(97)
num_frames = 24 # 16
clip_steps = 12
Bs_Test = 1

def prepare_hmdb51_test_loader(data_dir, split_dir, num_frames=24, clip_steps=12, batch_size=1):
    torch.manual_seed(97)
    transform_test_A5 = transforms.Compose([
                                 T.ToFloatTensorInZeroOne(),
                                 T.Resize((400, 400)),
                                 #T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                 T.CenterCrop((320, 320))]) #Size for model

    hmdb51_test = torchvision.datasets.HMDB51(data_dir, split_dir,
                                              frames_per_clip=num_frames, frame_rate=24,
                                              step_between_clips=num_frames, fold=1,
                                              train=False, transform=transform_test_A5,
                                              num_workers=2)

    test_loader = DataLoader(hmdb51_test, batch_size=Bs_Test, shuffle=False)

    return test_loader
```

1. function evaluate model + create file predict

```python
def evaluate_model(loaded_model, test_loader, output_path):
    device = next(loaded_model.parameters()).device
    true_labels, predicted_labels = [], []
    with torch.no_grad():
        for data, _, target in test_loader_A5:
            # Move data and target to the same device as the model
            data, target = data.to(device), target.to(device)

            output = F.log_softmax(loaded_model(data), dim=1)
            _, predicted = torch.max(output, dim=1)

            true_labels.extend(target.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    # Create DataFrame from true_labels and predicted_labels
    df_output = pd.DataFrame({'true_label': true_labels, 'predicted_label': predicted_labels})

    # Save DataFrame as a CSV file
    df_output.to_csv(output_path, index=False)
```

1. function combined csv (time + predict)

```python
def evaluate_model(loaded_model, test_loader, output_path):
    device = next(loaded_model.parameters()).device
    true_labels, predicted_labels = [], []
    with torch.no_grad():
        for data, _, target in test_loader_A5:
            # Move data and target to the same device as the model
            data, target = data.to(device), target.to(device)

            output = F.log_softmax(loaded_model(data), dim=1)
            _, predicted = torch.max(output, dim=1)

            true_labels.extend(target.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    # Create DataFrame from true_labels and predicted_labels
    df_output = pd.DataFrame({'true_label': true_labels, 'predicted_label': predicted_labels})

    # Save DataFrame as a CSV file
    df_output.to_csv(output_path, index=False)
```

### 14. how to use function

```python
for i in range(1,20): #range = how many video
    inputVideo = f"yor_path/รอบ {i}.avi"
    outputFolderVideo = f"yor_path/รอบ {i}/test"
    output_csv = f"yor_path/Round{i}.csv"
    output_file_text = f"yor_path/รอบ {i}/test_test_split1.txt"
    data_dir = f"yor_path/รอบ {i}/"
    split_dir = f"yor_path/รอบ {i}/"
    output_path = f"yor_path/Round{i}.csv"
    path3 = f"/yor_path/Predict+time_Round_{i}.csv"
    clipDuration = 2
    cut_video(inputVideo, outputFolderVideo, clipDuration, output_csv)
    df = pd.read_csv(output_csv)
    create_test_split(outputFolderVideo, output_file_text)
    test_loader_A5 = prepare_hmdb51_test_loader(data_dir, split_dir)
    print(test_loader_A5.dataset)
    evaluate_model(loaded_model, test_loader_A5, output_path)
    combine_and_transform_csv(output_path, output_csv, path3)
```