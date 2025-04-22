import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, accuracy_score, auc

from torch.nn import functional as F
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def calculate_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def main():
    args = parse.parse_args()
    test_path = args.test_path
    batch_size = args.batch_size
    torch.backends.cudnn.benchmark = True
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    test_dataset = torchvision.datasets.ImageFolder(test_path, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                              num_workers=8)

    test_dataset_size = len(test_dataset)

    model = torch.load("./checkpoint/multi_model_32.pt", map_location='cpu')
    model_2 = torch.load("./checkpoint/two_model_32.pt", map_location="cpu")

    model = model.cuda()
    model_2 = model_2.cuda()

    model.eval()
    model_2.eval()

    corrects = 0
    pred_all = []
    pro_all = []
    label_all = []
    F1_prob_all = []

    with torch.no_grad():
        for (image, labels) in test_loader:
            image = image.cuda()
            labels = labels.cuda()

            # FF++
            labels_replace0 = torch.zeros_like(labels)
            labels_replace1 = torch.ones_like(labels)
            labels_new = torch.where(labels != 0, labels, labels_replace1)
            labels = labels_new
            labels_new = torch.where(labels != 1, labels, labels_replace1)
            labels = labels_new
            labels_new = torch.where(labels != 2, labels, labels_replace1)
            labels = labels_new
            labels_new = torch.where(labels != 3, labels, labels_replace1)
            labels = labels_new
            labels_new = torch.where(labels != 4, labels, labels_replace0)
            labels = labels_new

            features_2 = model_2.features(image)
            features = model.features(image)

            _, outputs = model_2(features_2, features)
            _, preds = torch.max(outputs.data, 1)

            pred_all.extend(preds.cpu().numpy())

            outputs = F.softmax(outputs, dim=1)

            pro_all.extend(outputs[:, 1].cpu().numpy())
            F1_prob_all.extend(np.argmax(outputs.cpu().numpy(), axis=1))
            label_all.extend(labels.cpu().numpy())
            corrects += torch.sum(preds == labels.data).to(torch.float32)

            print('Iteration Acc {:.4f}'.format(torch.sum(preds == labels.data).to(torch.float32) / len(image)))

    print("ACC:{:.4f}".format(accuracy_score(label_all, pred_all)))
    print("AUC:{:.4f}".format(roc_auc_score(label_all, pro_all)))
    print("AP:{:.4f}".format(average_precision_score(label_all, pro_all)))
    print("EER:{:.4f}".format(calculate_eer(label_all, pro_all)))



if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--batch_size', type=int, default=256)
    parse.add_argument('--test_path', type=str, default=" ")
    main()
