from __future__ import print_function
import os
import numpy as np
from extract import PointCloudProject
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import PointNet, DGCNN
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
    all_labels_list = []

    # Iterate over each project name
    for project_name in project_names:
        project = PointCloudProject(project=project_name)
        box_pcd, labels = project.extract_box_and_label(resample=True, num_points=num_points)

        # Convert points from each point cloud in the list to numpy arrays
        points_np_list = [np.asarray(pcd.points) for pcd in box_pcd]

        # Add points and labels to the main lists
        all_points_np_list.extend(points_np_list)
        all_labels_list.extend(labels)

    # Concatenate all points into a single numpy array
    all_points_np = np.array(all_points_np_list)

    # Find unique labels, sort them, and create a mapping
    unique_labels = np.unique(all_labels_list)
    unique_labels.sort()
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

    print("The labels found:", label_to_int)

    # Map all_labels_list to integers
    all_labels_np = np.array([label_to_int[label] for label in all_labels_list])

    # Split the data into training and testing sets and shuffle
    X_train, X_test, y_train, y_test = train_test_split(all_points_np, all_labels_np, test_size=0.2, random_state=42, shuffle=True)

    return X_train, X_test, y_train, y_test


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

    # Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
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

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            # Print current loss and remaining batches
            print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.6f}")
        scheduler.step()
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_loss /= count
        train_acc = metrics.accuracy_score(train_true, train_pred)
        train_avg_acc = metrics.balanced_accuracy_score(train_true, train_pred)

        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                train_loss,
                                                                                train_acc,
                                                                                train_avg_acc)
                
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_loss /= count
        test_acc = metrics.accuracy_score(test_true, test_pred)
        test_avg_acc = metrics.balanced_accuracy_score(test_true, test_pred)

        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss,
                                                                              test_acc,
                                                                              test_avg_acc)
                  
                       
        io.cprint(outstr)
        
        # if test_acc >= best_test_acc:
        #     best_test_acc = test_acc
        #     # Save the model
        #     # torch.save(model.state_dict(), 'outputs_classification/%s/models/model.t7' % args.exp_name)
        #     torch.save(model.state_dict(), f'outputs_classification/{args.exp_name}/model_epoch_{epoch}.pth')

        
        # torch.save(model.state_dict(), f'outputs_classification/{args.exp_name}/model_epoch_{epoch}.pth')

        # Save losses and accuracies to file
        # with open(f'outputs_classification/{args.exp_name}/metrics.txt', 'a') as f:
        #     f.write(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f}, Train Avg Acc: {train_avg_acc:.6f}, Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.6f}, Test Avg Acc: {test_avg_acc:.6f}\n')

    # Save final model
    # torch.save(model.state_dict(), f'outputs_classification/{args.exp_name}/model_final.pth')

if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'],
                        help='Dataset to use, [modelnet40]')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args, unknown = parser.parse_known_args()

    io = IOStream('outputs_classification/%s/run.log' % args.exp_name)
    io.cprint(str(args))

    train(args, io)