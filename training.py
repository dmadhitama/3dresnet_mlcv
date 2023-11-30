from utils import AverageMeter, MeanAveragePrecision
from utils import Colors, print_color
from sklearn.metrics import average_precision_score
import numpy as np


def train_epoch(epoch,
                data_loader,
                model,
                criterion,
                optimizer,
                device):

    losses = AverageMeter()
    all_targets = []
    all_predictions = []

    model.train()
    for i, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        outputs_np = outputs.cpu().detach().numpy()
        targets_np = targets.cpu().detach().numpy()

        all_predictions.extend(outputs_np)
        all_targets.extend(targets_np)

        print_color(f'Training [{epoch}]\t | Batch {i}/{len(data_loader)}\t | Loss {losses.avg}', Colors.GREEN)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # After the epoch, calculate overall mAP
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute mAP using sklearn's average_precision_score
    epoch_mAP = average_precision_score(all_targets, all_predictions, average='macro')

    print_color(f'Training [{epoch}]\t | Overall mAP {epoch_mAP}', Colors.RED)
