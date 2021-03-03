#   1. You have to change the code so that he model is trained on the train set,
#   2. evaluated on the validation set.
#   3. The test set would be reserved for model evaluation by teacher.

from args import get_parser
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import os, random, math
from models import fc_model
from dataset import get_loader
import sys
from copy import copy
import matplotlib.pyplot as plt
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# -------------------------------------------

def draw_graph(train, validation, metric):
    epochs = list(range(1, len(train)+1))

    plt.plot(epochs, train, label = "train") 
     
    plt.plot(epochs, validation, label = "valid") 
    
    # naming the x axis 
    plt.xlabel('epochs') 
    # naming the y axis 
    plt.ylabel(metric) 
    # giving a title to my graph 
    plt.title(metric) 
    
    # show a legend on the plot 
    plt.legend() 
    
    # function to show the plot 
    #plt.show() 
    plt.savefig(metric + '.png')
    plt.clf()


def main(args):

    data_loader, dataset = get_loader(args.data_dir, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.num_workers, drop_last=False, args=args)

    
    val_args = copy(args)

    val_args.mode = "valid"

    val_data_loader, _ = get_loader(args.data_dir, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.num_workers, drop_last=False, args=val_args)
    


    data_size = dataset.get_data_size()
    num_classes = dataset.get_num_classes()
    instance_size = dataset.get_instance_size()

    confusion_matrix = [[0] * num_classes] * num_classes
    
    
    # Build the model
    model = fc_model(input_size=instance_size, num_classes=num_classes, dropout=args.dropout)
    
    if torch.cuda.is_available() == False:
        summary(
            model, input_size=(3, 224, 224), batch_size=args.batch_size, device='cuda')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    
    print(model)

    # create optimizer
    params = list(model.parameters())

    optimizer = torch.optim.SGD(params, momentum=0.9, lr=args.learning_rate)
    label_crit = nn.CrossEntropyLoss()

    model = model.to(device)
    

    print ("model created & starting training ...\n\n")
    
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []
    best_acc = 0.0
    lr = args.learning_rate
    # Training script
    for epoch in range(args.num_epochs):
        model.train()
        total_correct_preds = 0.0
        total = 1e-10
        loss = 0.0
        if (epoch+1) % 60 == 0:
            
            lr = lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        #step loop
        for step, (image_input, class_idxs, _) in enumerate(data_loader):

            # move all data loaded from dataloader to gpu
            class_idxs = class_idxs.to(device)
            image_input = image_input.to(device)

            # feed-forward data in the model
            output = model(image_input) # 32 * 150528 --> 32 * 11

            # compute losses
            state_loss = label_crit(output, class_idxs) # --> 32 * 1

            # aggregate loss for logging
            loss += state_loss.item() * output.size(0)

            # back-propagate the loss in the model & optimize
            model.zero_grad()
            state_loss.backward()
            optimizer.step()

            # accuracy computation
            _, pred_idx = torch.max(output, dim=1)
            total_correct_preds += torch.sum(pred_idx==class_idxs).item()
            total += output.size(0)

        # epoch accuracy & loss
        accuracy = round(total_correct_preds/total, 2)
        loss = round(loss/total, 3)
        train_loss.append(loss)
        train_accuracy.append(accuracy)

        print('\nepoch {}: For training: total_correct_preds: {}, total: {},  accuracy: {}, loss: {}'.format(epoch, total_correct_preds, total, accuracy, loss), end="")

        # you can save the model here at specific epochs (ckpt) to load and evaluate the model on the val set

        model.eval()
        eval_loss = 0.0
        total_correct_preds = 0.0
        total = 1e-10
        

        with torch.no_grad():
            for step, (image_input, class_idxs, _) in enumerate(val_data_loader):
                # move all data loaded from dataloader to gpu
                class_idxs = class_idxs.to(device)
                image_input = image_input.to(device)

                # feed-forward data in the model
                output = model(image_input)

                # compute losses
                state_loss = label_crit(output, class_idxs)

                # aggregate loss for logging
                eval_loss += state_loss.item() * output.size(0)

                # accuracy computation
                _, pred_idx = torch.max(output, dim=1)
                total_correct_preds += torch.sum(pred_idx==class_idxs).item()
                total += output.size(0)

            accuracy = round(total_correct_preds/total, 2)
            eval_loss = round(eval_loss/total, 3)
            print('\nepoch {}: For validation: total_correct_preds: {}, total: {},  accuracy: {}, loss: {}'.format(epoch, total_correct_preds, total, accuracy, eval_loss), end="")

            val_loss.append(eval_loss)
            val_accuracy.append(accuracy)
            if accuracy > best_acc:
                best_acc = accuracy
                best_model = model
                print('\nfound new best accuracy: ', best_acc)


        if epoch > 50 and epoch % 7 == 0:
            checkpoint_path = "checkpoints/checkpoint{}.pt".format(epoch)
            torch.save(model.state_dict(), checkpoint_path)


    torch.save(best_model.state_dict(), "checkpoint.pt")

    print()
    
    draw_graph(train_loss, val_loss, 'loss')
    draw_graph(train_accuracy, val_accuracy, 'accuracy')
    


if __name__ == '__main__':
    args = get_parser()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    main(args)