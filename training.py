from utils import AverageMeter
from utils import Colors, print_color, get_lr
import time

def train_epoch(epoch,
                data_loader,
                model,
                criterion,
                optimizer,
                device,
                tb_writer):
    model.train()

    losses = AverageMeter()
    current_lr = get_lr(optimizer)

    for i, (inputs, targets) in enumerate(data_loader):

        start_time = time.time()
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        losses.update(loss.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end_time = time.time()
        print_color(f'Training {epoch:2d} | Step [{i:{len(str(len(data_loader)))}}/{len(data_loader)}] | Elapsed Time {end_time - start_time:.3f} | Lr {current_lr} | Loss {losses.avg:.5f}', Colors.GREEN)

        if tb_writer is not None:
            tb_writer.add_scalar('train/batch/loss', losses.avg, i)

    if tb_writer is not None:
        tb_writer.add_scalar('train/epoch/loss', losses.avg, epoch)
        tb_writer.add_scalar('train/epoch/lr', current_lr, epoch)
