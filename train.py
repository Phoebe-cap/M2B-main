import argparse
import os
import torchvision
from balanced_loss import Loss

import torch
from torchvision import transforms

from networks.xception_fusion import Xception_fusion
from networks.am_softmax import AMSoftmaxLoss

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--train_folder', type=str,default=" ")
    parser.add_argument('--val_folder', type=str, default=" ")
    parser.add_argument('--out_folder', type=str, default=" ")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoches', type=int, default=32)
    opt = parser.parse_args()
    return opt

def main(opt):

    ckp_old = torch.load("./xception-b5690688.pth",map_location="cpu")
    ckp={}
    for name, weights in ckp_old.items():
        if not name.startswith("fc"):
            ckp[name] = weights
            if 'pointwise' in name:
                ckp[name] = weights.unsqueeze(
                    -1).unsqueeze(-1)

    samples_per_class = [ ]
    # criterion_two = torch.nn.CrossEntropyLoss()
    criterion_two = AMSoftmaxLoss(gamma=0., m=0.45, s=30, t=1.)
    criterion_multi = Loss(
        loss_type="focal_loss",
        samples_per_class=samples_per_class,
        class_balanced=True
    )

    multi = Xception_fusion(num_classes=5)
    multi.load_state_dict(ckp, strict=False)
    multi = multi.cuda()
    two = Xception_fusion(num_classes=2)
    two.load_state_dict(ckp, strict=False)
    two = two.cuda()

    multi_optimizer = torch.optim.Adam(multi.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    multi_scheduler = torch.optim.lr_scheduler.StepLR(multi_optimizer, step_size=5, gamma=0.5)
    two_optimizer = torch.optim.Adam(two.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
    two_scheduler = torch.optim.lr_scheduler.StepLR(two_optimizer, step_size=5, gamma=0.5)

    train_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    train_dataset = torchvision.datasets.ImageFolder(opt.train_folder, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                               drop_last=False, num_workers=8)
    val_dataset = torchvision.datasets.ImageFolder(opt.val_folder, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True,
                                               drop_last=False, num_workers=8)
    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)

    # Train the model using multiple GPUs
    # model = torch.nn.DataParallel(model)

    for epoch in range(opt.epoches):
        print('Epoch {}/{}'.format(epoch + 1, opt.epoches))
        print('-' * 10)
        multi.train()
        two.train()
        epoch_loss = []
        epoch_multi_loss = []
        epoch_two_loss = []
        epoch_label_loss = []
        val_loss = 0.0
        val_corrects = 0.0
        pos_function = torch.nn.Sigmoid()
        iteration = 0
        if epoch < opt.epoches // 2:
            for (image, labels) in train_loader:
                image = image.cuda()
                labels = labels.cuda()

                fea_two = two.features(image)
                fea_multi = multi.features(image)
                _, out_two = two(fea_two, fea_multi)
                _, out_multi = multi(fea_multi, fea_two)

                loss_multi = criterion_multi(out_multi, labels)

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

                loss_two = criterion_two(out_two, labels)

                out_two = pos_function(out_two).squeeze()
                _, preds_two = torch.max(out_two.data, 1)

                out_multi = pos_function(out_multi).squeeze()
                _, preds_multi = torch.max(out_multi.data, 1)

                preds_multi_replace0 = torch.zeros_like(preds_multi)
                preds_multi_replace1 = torch.ones_like(preds_multi)
                preds_multi_new = torch.where(preds_multi > 3, preds_multi_replace0, preds_multi_replace1)
                preds_multi = preds_multi_new

                loss_label = abs(preds_two - preds_multi).float().mean()

                loss = loss_two + 0.1 * loss_multi + loss_label

                epoch_loss.append(loss.item())
                epoch_label_loss.append(loss_multi.item())
                epoch_multi_loss.append(loss_multi.item())
                epoch_two_loss.append(loss_two.item())

                multi_optimizer.zero_grad()
                two_optimizer.zero_grad()
                loss.backward()
                multi_optimizer.step()
                two_optimizer.step()

                iteration += 1
                if iteration % 100 == 0:
                   print('loss: {:.4f}  multi_loss: {:.4f}  two_loss: {:.4f}  label_loss: {:.4f}  '.format(loss, 0.1*loss_multi, loss_two, loss_label))
            two_scheduler.step()
            multi_scheduler.step()

        else:
            multi.eval()
            multi.requires_grad_(False)
            for (image, labels) in train_loader:
                image = image.cuda()
                labels = labels.cuda()

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

                fea_two = two.features(image)
                fea_multi = multi.features(image)
                _, out_two = two(fea_two, fea_multi)

                loss_two = criterion_two(out_two, labels)

                loss = loss_two

                epoch_loss.append(loss.item())
                epoch_two_loss.append(loss_two.item())
                two_optimizer.zero_grad()
                loss.backward()
                two_optimizer.step()

                if iteration % 100 == 0:
                    print('loss: {:.4f} '.format(loss))
            two_scheduler.step()

        multi.eval()
        two.eval()

        with torch.no_grad():
            for (image, labels) in val_loader:
                image = image.cuda()
                labels = labels.cuda()

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

                fea_two = two.features(image)
                fea_multi = multi.features(image)
                _, out_two = two(fea_two, fea_multi)
                out_two = pos_function(out_two).squeeze()
                _, preds = torch.max(out_two.data, 1)

                val_corrects += torch.sum(preds == labels.data).to(torch.float32)

            epoch_acc = val_corrects / val_dataset_size

            with open(os.path.join(opt.out_folder, 'val.log'), 'a') as log:
                log.write(str(epoch + 1) + '\t' + 'val_acc:' + str(epoch_acc) + '\t' + '\n')
            print('val_acc: {:.4f} '.format(epoch_acc))

            torch.save(multi, os.path.join(opt.out_folder, 'checkpoint', 'multi_model_%d.pt' % (epoch + 1)))
            torch.save(two, os.path.join(opt.out_folder, 'checkpoint', 'two_model_%d.pt' % (epoch + 1)))

        two_scheduler.step()
        multi_scheduler.step()

    return two

if __name__=='__main__':
    opt = parse_option()
    main(opt)