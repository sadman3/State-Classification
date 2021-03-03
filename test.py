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
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

id_to_class = {
    "0": "creamy_paste",
    "1": "diced",
    "2": "floured",
    "3": "grated",
    "4": "juiced",
    "5": "jullienne",
    "6": "mixed",
    "7": "other",
    "8": "peeled",
    "9": "sliced",
    "10": "whole"
}

result_file_name = "sadman_result.json"


path_to_class = {}
def prepare_json_result(image_paths, predictions):
    for i in range(len(image_paths)):
        path_separated = image_paths[i].split("/")
        path = path_separated[2] + '/' + path_separated[3]
        class_name = id_to_class[str(predictions[i].item())]
        path_to_class[path] = class_name
    


def main(args):
    args.mode = "test"

    test_data_loader, dataset = get_loader(args.data_dir, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.num_workers, drop_last=False, args=args)

    num_classes = dataset.get_num_classes()
    instance_size = dataset.get_instance_size()

    ckpt = []
    for _file in os.listdir("checkpoints"):
        
        model = fc_model(input_size=instance_size, num_classes=11, dropout=args.dropout)
        
        model.load_state_dict(torch.load("checkpoints/" + _file, map_location=lambda storage, loc: storage))
        model = model.to(device)
        model.eval()
        ckpt.append(model)

    # total_correct_preds = 0.0
    # total = 1e-10

    with torch.no_grad():
        for step, (image_input, class_idxs, path) in enumerate(test_data_loader):
            # move all data loaded from dataloader to gpu
            class_idxs = class_idxs.to(device)
            image_input = image_input.to(device)

            result = torch.zeros(num_classes)
            result = result.to(device)

            for model in ckpt:
                # feed-forward data in the model
                output = model(image_input)

                result = result.add(output)

            # accuracy computation
            _, pred_idx = torch.max(result, dim=1)

            # total_correct_preds += torch.sum(pred_idx==class_idxs).item()
            # total += output.size(0)


            prepare_json_result(path, pred_idx)
        # accuracy = round(total_correct_preds/total, 2)
    
        # print('total_correct_preds: {}, total: {},  accuracy: {}\n'.format(total_correct_preds, total, accuracy), end="")

        json.dump(path_to_class, open(result_file_name, 'w'))


if __name__ == '__main__':
    args = get_parser()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    main(args)


