
import sys
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
from models import Model
from torch.nn import functional as F
import data_utils
from logger import get_logger
import matplotlib.pyplot as plt



# training function
def train(model, optimizer, scheduler):
    
    ################################################################################
    Test_best_acc= 0.0
    for i in range(args.epochs):
        t = time.time()
        total_loss = []
        total_train_acc = []
        total_imp_loss = []
        total_cls_loss = []
        model.train()
        for input, prediction_target, mask, label_target, batch_size, batch_need_label in data_utils.next_batch(args.batch_size, train_data, train_label, True,args.input_dimension_size,args.seq_len, Trainable = True):
            input = torch.FloatTensor(input).to(args.device)
            prediction_target = torch.FloatTensor(prediction_target).to(args.device)
            prediction_target = torch.reshape(input=prediction_target,shape=[-1, args.input_dimension_size])
            mask = torch.FloatTensor(mask).to(args.device)
            mask = torch.reshape(input=mask,shape=[-1, args.input_dimension_size])
            label_target = torch.FloatTensor(label_target).to(args.device)
            label = torch.LongTensor(batch_need_label).to(args.device)
            optimizer.zero_grad()
            logits, prediction = model(input)

            loss_imp = torch.mean(torch.square( (prediction_target - prediction) * mask) ) / (args.batch_size)
            loss_cls = criterion(logits, label)
            loss = loss_cls + loss_imp
            loss.backward()
            optimizer.step()
            # acc
            label_predict = F.softmax(logits,dim=1)
            correct_predictions = torch.argmax(label_predict, dim=1) == torch.argmax(label_target, dim=1)  
            accuracy = correct_predictions.float() #float32
            total_train_acc.append(accuracy.data.cpu().numpy())
            total_imp_loss.append(loss_imp.item())
            total_cls_loss.append(loss_cls.item())
            total_loss.append(loss.item())
        scheduler.step() 
        Loss = np.mean(total_loss)
        Loss_imp = np.mean(total_imp_loss)
        Loss_cls = np.mean(total_cls_loss)
        Train_acc = np.mean(np.array(total_train_acc).reshape(-1))
        t2 = time.time() - t

        #----test----

        test_acc = []
        total_sample_num = 0
        model.eval()
        for input, prediction_target, mask, label_target, batch_size, batch_need_label in data_utils.next_batch(args.batch_size, test_data, test_label, True,args.input_dimension_size,args.seq_len, Trainable = False):
            input = torch.FloatTensor(input).to(args.device)
            label_target = torch.FloatTensor(label_target).to(args.device)
            out, _  = model(input)
            label_predict = F.softmax(out,dim=1)
            correct_predictions = torch.argmax(label_predict, dim=1) == torch.argmax(label_target, dim=1)  
            accuracy = correct_predictions.float() #float32
            total_sample_num += batch_size
            test_acc.append(accuracy.data.cpu().numpy())
        Test_acc = np.mean(np.array(test_acc).reshape(-1)[:total_sample_num])

        if(Test_acc > Test_best_acc):
            Test_best_acc = Test_acc


        print('Epoch[{:4d}/{:4d}]'.format(i + 1, args.epochs),
              'Loss_train: {:.4f}'.format(Loss),
              'Loss_IMP:{:.4f}'.format(Loss_imp),
              'Loss_CLS:{:.4f}'.format(Loss_cls),
              'acc_train: {:.4f}'.format(Train_acc),
              'acc_val: {:.4f}'.format(Test_acc),
              'best_val_acc: {:.4f}'.format(Test_best_acc),
              'time: {:.2f}s'.format(t2))

        logger.info('Epoch:[{}/{}] ' ' val_acc={:.4f} '
                    ' best_acc={:.4f} '.format(
                        i + 1, args.epochs, Test_acc, Test_best_acc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default='16')
    parser.add_argument('--d_model', type=int, default='128')  

    parser.add_argument('--epochs', type=int, default='50')
    parser.add_argument('--filename', type=str, default='Adiac')
    parser.add_argument('--missing_ratio', type=int, default='20')  
    parser.add_argument('--learning_rate', type=float, required=False, default=3e-4) 
    parser.add_argument('--GPU', type=str, default='0')

    # cuda settings
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    parser.add_argument('--wd', type=float, default=5e-3,
                        help='Weight decay (L2 loss on parameters). default: 5e-3')

    args = parser.parse_args()
    args.device = torch.device("cuda:" + args.GPU  if torch.cuda.is_available() else "cpu")
    print("device using:", args.device)

    train_data_path = '../results/data/' + args.filename + '/' + args.filename + '_TRAIN_' + str(args.missing_ratio) + '.csv'
    test_data_path = '../results/data/' + args.filename + '/' + args.filename + '_TEST_' + str(args.missing_ratio) + '.csv'

    print ('Loading data && Transform data--------------------')
    print (train_data_path)
    train_data, train_label = data_utils.load_data(train_data_path)
    #For univariate
    args.seq_len = train_data.shape[1] 
    args.input_dimension_size = 1  

    print("nums_steps:",args.seq_len)

    train_label, num_classes = data_utils.transfer_labels(train_label) #when without noise label open this line
    args.num_class = num_classes
    print("train label(transfer):",train_label)


    print ('Train Label:', np.unique(train_label))
    print ('Train data completed-------------')

    test_data, test_labels = data_utils.load_data(test_data_path)

    test_label, test_classes = data_utils.transfer_labels(test_labels)
    print ('Test data completed-------------')

    #logger = get_logger('./ret/ours_' + args.filename + '_' +  str(args.missing_ratio) + '.log')
    model = Model(out=args.num_class,seq_len=args.seq_len, dmodel=args.d_model, device=args.device).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)

    train(model, optimizer, scheduler)


