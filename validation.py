import torch
from utils import AverageMeter
from utils import Colors, print_color
import time
from sklearn.metrics import average_precision_score
import numpy as np

def val_epoch(epoch,
              data_loader,
              model,
              criterion,
              device,
              tb_writer):

    model.eval()

    losses = AverageMeter()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        
        for i, (inputs, targets) in enumerate(data_loader):
            start_time = time.time()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))

            all_predictions.extend(outputs.cpu().detach().numpy())
            all_targets.extend(targets.cpu().detach().numpy())

            end_time = time.time()
            print_color(f'Validating {epoch:2d} | Step [{i:{len(str(len(data_loader)))}}/{len(data_loader)}] | Elapsed Time {end_time - start_time:.3f} | Loss {losses.avg:.5f}', Colors.YELLOW)

            if tb_writer is not None:
                tb_writer.add_scalar('val/batch/loss', losses.avg, i)

        # After the epoch, calculate overall mAP
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Compute mAP using sklearn's average_precision_score
        epoch_mAP = average_precision_score(all_targets, all_predictions, average='macro')

        print_color(f'Validating {epoch:3} | mAP {epoch_mAP:5f}', Colors.BLUE)

        if tb_writer is not None:
            tb_writer.add_scalar('val/epoch/loss', losses.avg, epoch)
            tb_writer.add_scalar('val/epoch/map', epoch_mAP, epoch)

    return losses.avg
