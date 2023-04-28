import os

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from model import INet
from dataset import Dataset
from torch import nn
import numpy as np
import pandas as pd
from PIL import Image
import json
from torchvision.transforms import *

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def validate(network, validation_set):
    """
  This function validates convnet parameter optimizations
  """
    #  creating a list to hold loss per batch
    loss_per_batch = []

    #  defining model state
    network.eval()

    #  defining dataloader
    val_loader = DataLoader(
        validation_set,
        batch_size=32,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    print('validating...')
    criterion = nn.CrossEntropyLoss().cuda()

    #  instantiating counters
    total_correct = 0
    total_instances = 0

    #  preventing gradient calculations since we will not be optimizing
    with torch.no_grad():
        #  iterating through batches
        iterator = tqdm(val_loader, total=len(val_loader))
        for i, (images, labels) in enumerate(iterator):
            # --------------------------------------
            #  sending images and labels to device
            # --------------------------------------
            images, labels = images.cuda(), labels.cuda().float()

            # --------------------------
            #  making classsifications
            # --------------------------
            classifications = network(images)
            # print('classifications', classifications)

            # -----------------
            #  computing loss
            # -----------------
            loss = criterion(classifications, labels).float()
            loss_per_batch.append(loss.item())

            classifications = torch.argmax(classifications, dim=1)
            labels = torch.argmax(labels, dim=1)
            # print('classifications, index', classifications)
            # --------------------------------------------------
            #  comparing indicies of maximum values and labels
            # --------------------------------------------------
            correct_predictions = sum(classifications == labels).item()

            # ------------------------
            #  incrementing counters
            # ------------------------
            total_correct += correct_predictions
            total_instances += len(images)

    avg_loss = np.average(loss_per_batch)
    avg_acc = round(total_correct / total_instances, 3)
    print('all done!')
    print('average loss', avg_loss)
    print('accuracy', avg_acc)

    return avg_loss, avg_acc

# effect: "", ""
def model_predict_json(network, effect):
    network.eval()

    predict_list = []
    img_dir = '../dataset/frames%s/' % effect
    results_path = 'results/frames%s.json' % effect

    transform = Compose([
        Resize((112, 112)),
        ToTensor(),
        Normalize([0.485], [0.229])
    ])

    with torch.no_grad():
        for name in os.listdir(img_dir):
            image_id = name[:-4]
            image = transform(Image.open(img_dir + name).convert('RGB')).unsqueeze(0).cuda()
            effect_prob = network(image)
            dic = {'image_id': image_id, 'effect_prob': list(effect_prob.cpu().numpy())[0]}
            predict_list.append(dic)

    with open(results_path, 'w') as f:
        json.dump(predict_list, f, cls=NpEncoder)

def model_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_path = './log/2023-04-08_12-40/checkpoints/model_{epoch+1:04d}.pth'
    model = INet().to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    valid_dataset = Dataset('../dataset/val2014_classifier/')

    # write to csv
    df = pd.DataFrame({
        "evaluation_metrics": ['valid set avg_loss', 'valid set avg_acc'],
        "value": validate(model, valid_dataset)
    })
    df.to_csv(weight_path[:-4] + '.csv')
    print(df)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_path = './log/2023-04-08_12-40/checkpoints/model_{epoch+1:04d}.pth'
    model = INet().to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model_predict_json(model, "_dark")

    # model_predict_json(model, "_random1k_resolution")
