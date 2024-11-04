from __future__ import print_function
import os
import numpy as np
from extract import PointCloudProject
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from model import DGCNN_semseg_s3dis
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from util import cal_loss, IOStream
import sklearn.metrics as metrics


def load_data(num_points):
    # Path to the training set folders
    training_set_path = 'data/TrainingSet'

    # Find all folder names in the training set directory
    project_names = [f for f in os.listdir(training_set_path) if os.path.isdir(os.path.join(training_set_path, f))]

    print("Project names:", project_names)

    all_points_np_list = []
    all_points_labels_list = []

    # Iterate over each project name
    for project_name in project_names[2:3]:
        project = PointCloudProject(project=project_name)
        box_pcd, points_labels = project.extract_box_and_points_labels(resample=True, num_points=num_points)

        # Convert points from each point cloud in the list to numpy arrays
        points_np_list = [np.asarray(pcd.points) for pcd in box_pcd]

        # Add points and labels to the main lists
        all_points_np_list.extend(points_np_list)
        all_points_labels_list.extend(points_labels)

    # Concatenate all points into a single numpy array
    all_points_np = np.array(all_points_np_list)

    # # Find unique labels, sort them, and create a mapping
    # unique_labels = np.unique(all_points_labels_list)
    # unique_labels.sort()
    # label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

    # print("The labels found:", label_to_int)

    all_points_labels_np = np.array(all_points_labels_list)

    # Split the data into training and testing sets and shuffle
    X_train, X_test, y_train, y_test = train_test_split(all_points_np, all_points_labels_np, test_size=0.2, random_state=42, shuffle=True)

    return X_train, X_test, y_train, y_test

def calculate_sem_IoU(pred_np, seg_np, visual=False):
    I_all = np.zeros(2)
    U_all = np.zeros(2)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(2):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all 

def train(args, io):
    # Load data using the new data loading function
    X_train, X_test, y_train, y_test = load_data(num_points=args.num_points)

    # Print the shape of the data to determine num_features
    print("Training points shape:", X_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Testing points shape:", X_test.shape)
    print("Testing labels shape:", y_test.shape)

    # Ensure k is not greater than the number of points
    num_points = X_train.shape[1]
    num_features = X_train.shape[2]  # Determine num_features
    print(f"Number of features: {num_features}")

    if args.k > num_points:
        print(f"Warning: k ({args.k}) is greater than the number of points ({num_points}). Adjusting k to {num_points}.")
        args.k = num_points

    # Set environment variables for debugging and multiprocessing
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1' 

    # device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() and not args.no_cuda else "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Move data to the appropriate device and convert to float32
    X_train, y_train = (
        torch.tensor(X_train, dtype=torch.float32).to(device),
        torch.tensor(y_train, dtype=torch.int64).to(device)
    )
    X_test, y_test = (
        torch.tensor(X_test, dtype=torch.float32).to(device),
        torch.tensor(y_test, dtype=torch.int64).to(device)
    )

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=0)


    # train_loader = DataLoader(S3DIS(partition='train', num_points=args.num_points, test_area=args.test_area), 
    #                           num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # test_loader = DataLoader(S3DIS(partition='test', num_points=args.num_points, test_area=args.test_area), 
    #                         num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("mps" if torch.backends.mps.is_available() and not args.no_cuda else "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")


    #Try to load models
    if args.model == 'dgcnn':
        model = DGCNN_semseg_s3dis(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.5, args.epochs)

    criterion = cal_loss

    best_test_iou = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        for batch_idx, (data, seg) in enumerate(train_loader):
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, 2), seg.view(-1,1).squeeze())
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            # Print current loss and remaining batches
            print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.6f}")
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        train_avg_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch, 
                                                                                                  train_loss*1.0/count,
                                                                                                  train_acc,
                                                                                                  train_avg_acc,
                                                                                                  np.mean(train_ious))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        for data, seg in test_loader:
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, 2), seg.view(-1,1).squeeze())
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        test_avg_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                              test_loss*1.0/count,
                                                                                              test_acc,
                                                                                              test_avg_acc,
                                                                                              np.mean(test_ious))
        io.cprint(outstr)
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            # torch.save(model.state_dict(), f'outputs_semantic_segmentation/models/{args.exp_name}/model_epoch_{epoch}.pth')

        # Save losses and accuracies to file
        with open(f'outputs_semantic_segmentation/{args.exp_name}/metrics.txt', 'a') as f:
            f.write(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f}, Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.6f}\n')

    # Save final model
    torch.save(model.state_dict(), f'outputs_semantic_segmentation/{args.exp_name}/model_final.pth')

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='S3DIS', metavar='N',
                        choices=['S3DIS'])
    parser.add_argument('--test_area', type=str, default=None, metavar='N',
                        choices=['1', '2', '3', '4', '5', '6', 'all'])
    parser.add_argument('--batch_size', type=int, default=3, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_root', type=str, default='', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--visu', type=str, default='',
                        help='visualize the model')
    parser.add_argument('--visu_format', type=str, default='ply',
                        help='file format of visualization')
    args = parser.parse_args()

    io = IOStream('outputs_semantic_segmentation/%s/run.log' % args.exp_name)
    io.cprint(str(args))

    train(args, io)