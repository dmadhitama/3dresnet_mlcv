import torch
from utils import AverageMeter
from utils import Colors, print_color

def val_epoch(epoch,
              data_loader,
              model,
              criterion,
              device):
    
    losses = AverageMeter()
    mAP = AverageMeter()
    model.eval()

    with torch.no_grad():
        
        for i, (inputs, targets) in enumerate(data_loader):

            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))

            print_color(f'Validation [{epoch}]\t| Batch {i}/{len(data_loader)}\t | Loss {losses.avg}', Colors.BLUE)

    return losses.avg
