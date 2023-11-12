import time
from .learning_utils import AverageMeter, ProgressMeter
import torch
def validate(valid_loader,
            model,
            criterion,
            metric_func,
            epoch,
            print_fn = print,
            batch_x_transformations = None,
            batch_y_transformations = None,
            meters = None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracies = AverageMeter('acc', ':6.2f')
    progress = ProgressMeter(len(valid_loader),
        [batch_time, data_time, losses, accuracies],
        prefix="Validation Epoch: [{}]".format(epoch),
        print_fn = print_fn)
    if meters is not None:
        for meter in meters:
            meter.reset()
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (batch_x, batch_y, batch_ident) in enumerate(valid_loader):
            if isinstance(batch_x,tuple) or isinstance(batch_x,list):
                if isinstance(batch_x[0], torch.Tensor):
                    batch_size = batch_x[0].shape[0]
                if isinstance(batch_x[0], list):
                    batch_size = len(batch_x[0])
            else:
                batch_size = batch_x.shape[0]
            if batch_x_transformations is not None:
                batch_x = batch_x_transformations(batch_x)
            if batch_y_transformations is not None:
                batch_y = batch_y_transformations(batch_y)
            data_time.update(time.time() - end)
            y_hat = model(batch_x)
            loss = criterion(y_hat, batch_y)
            accu, numel = metric_func(y_hat, batch_y)
            losses.update(loss.item(), batch_size)
            accuracies.update(accu.item(), numel)
            batch_time.update(time.time() - end)
            if meters is not None:
                for meter in meters:
                    meter(y_hat, batch_y, batch_ident)
            end = time.time()
        if meters is not None:
            for meter in meters:
                meter.display()
        progress.display(i+1)
    return accuracies.avg, losses.avg
def train (train_loader,
        model,
        criterion,
        metric_func,
        optimizer,
        epoch,
        batch_lr_scheduler = None,
        epoch_lr_scheduler = None,
        print_fn = print,
        batch_x_transformations = None,
        batch_y_transformations = None,
        meters = None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracies = AverageMeter('acc', ':6.2f')
    progress = ProgressMeter(len(train_loader),
        [batch_time, data_time, losses, accuracies],
        prefix="Training Epoch: [{}]".format(epoch),
        print_fn=print_fn)
    if meters is not None:
        for meter in meters:
            meter.reset()
    model.train()
    end = time.time()
    for i, (batch_x, batch_y, batch_ident) in enumerate(train_loader):
        if isinstance(batch_x,tuple) or isinstance(batch_x,list):
            if isinstance(batch_x[0], torch.Tensor):
                batch_size = batch_x[0].shape[0]
            if isinstance(batch_x[0], list):
                batch_size = len(batch_x[0])
        else:
            batch_size = batch_x.shape[0]
        if batch_x_transformations is not None:
            batch_x = batch_x_transformations(batch_x)
        if batch_y_transformations is not None:
            batch_y = batch_y_transformations(batch_y)
        data_time.update(time.time() - end)
        y_hat = model(batch_x)
        loss = criterion(y_hat, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_lr_scheduler:
            batch_lr_scheduler.step()
        accu, numel= metric_func(y_hat, batch_y)
        losses.update(loss.item(), batch_size)
        accuracies.update(accu.item(), numel)
        batch_time.update(time.time() - end)
        if meters is not None:
            for meter in meters:
                meter(y_hat, batch_y, batch_ident)
        end = time.time()
    if meters is not None:
        for meter in meters:
            meter.display()
    progress.display(i)
    if epoch_lr_scheduler:
        epoch_lr_scheduler.step()
    return accuracies.avg, losses.avg
def infer(test_loader, model,
        batch_x_transformations = None,
        batch_y_transformations = None,
        meters = None):
    if meters is not None:
        for meter in meters:
            meter.reset()
    res = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_ident in test_loader:
            if batch_x_transformations is not None:
                batch_x = batch_x_transformations(batch_x)
            if batch_y_transformations is not None:
                batch_y = batch_y_transformations(batch_y)
            y_hat = model(batch_x)
            res.append(y_hat)
            if meters is not None:
                for meter in meters:
                    meter(y_hat, batch_y, batch_ident)
        if meters is not None:
            for meter in meters:
                meter.display()
        return res
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
def find_lr(train_loader,
            model,
            criterion,
            optimizer,
            lr_init = 1e-8,
            lr_final = 10.,
            beta = 0.98,
            batch_x_transformations = None,
            batch_y_transformations = None):
    # number of learning rates increments to try
    num = len(train_loader) - 1
    #Get the value to multiply the lr with after each batch --> lr[i+1] = mult*lr[i]
    mult = (lr_final/lr_init)**(1/num)
    # Set lr to the initial value
    lr = lr_init
    # set the optimizer lr
    optimizer.param_groups[0]['lr'] = lr
    # exponentially weighted loss
    avg_loss = 0.
    # The best_loss attained so far
    best_loss = 0.
    # A counter maintaining the current batch
    batch_num = 0 
    # a list of smoothed_losses obtained by varying the learning rate.
    smoothed_losses = []
    # a list of losses without any smoothing
    raw_losses = []
    # the log of the learning rates for plotting
    lrs = []
    # put the model in the training mood
    model.train()
    # Iterate across batches
    for batch_x, batch_y, batch_ident in train_loader:
        if batch_x_transformations is not None:
            batch_x = batch_x_transformations(batch_x)
        if batch_y_transformations is not None:
            batch_y = batch_y_transformations(batch_y)
        batch_num += 1
        # Set the gradnients to zero
        optimizer.zero_grad()
        # Get model output
        y_hat = model(batch_x)
        # Calculate the loss for a batch
        loss = criterion(y_hat, batch_y)
        # Calculate the exponentially weighted loss
        avg_loss = beta * avg_loss + (1-beta)*loss.item()
        # Get the smoothed loss by averaging
        smoothed_loss = avg_loss/(1- beta**batch_num)
        # Early termination in case the smoothed_loss is way worse than the best (4X)
        if batch_num > 1 and smoothed_loss> 4*best_loss:
            return lrs, smoothed_losses, raw_losses
        # update the best_loss with the new minimum
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        # append the obtained smoothed and raw loss
        smoothed_losses.append(smoothed_loss)
        raw_losses.append(loss.item())
        # append the lr
        lrs.append(lr)
        # perform an optimization step
        loss.backward()
        optimizer.step()
        # update the lr
        lr *= mult
        # set the optimizer lr
        optimizer.param_groups[0]['lr'] = lr
    return lrs, smoothed_losses, raw_losses
class no_further_epochs (Exception):
    pass
def train_epoch(max_no_epochs,
                model,
                train_loader,
                valid_loader,
                criterion,
                optimizer,
                metric,
                batch_lr_scheduler = None,
                epoch_lr_scheduler = None,
                batch_x_transformations = None,
                batch_y_transformations = None,
                snap_shot_location = 'model.pt',
                resume = False,
                print_fn = print,
                valid_meters = None,
                train_meters = None):
    if "current_epoch" not in train_epoch.__dict__:
        train_epoch.current_epoch = 0
        train_epoch.best_attained_metric = -1
    if resume and "already_resumed" not in train_epoch.__dict__:
        snap_shot = torch.load(snap_shot_location, map_location = "cuda:0")
        model.load_state_dict(snap_shot['model'])
        train_epoch.best_attained_metric = snap_shot['best_attained_metric']
        print_fn(f"Restoring snapshot that attained results over validation data {train_epoch.best_attained_metric}")
        train_epoch.already_resumed = True
        train_epoch.current_epoch = 0
        validate(
                valid_loader, model,
                criterion, metric,
                train_epoch.current_epoch,
                print_fn = print_fn,
                batch_x_transformations = batch_x_transformations,
                batch_y_transformations = batch_y_transformations,
                meters = valid_meters)
    if train_epoch.current_epoch < max_no_epochs:
        train_avg_metric, train_avg_criterion = train(
                train_loader, model,
                criterion, metric, optimizer,
                train_epoch.current_epoch,
                batch_lr_scheduler = batch_lr_scheduler,
                epoch_lr_scheduler= epoch_lr_scheduler,
                print_fn = print_fn,
                batch_x_transformations = batch_x_transformations,
                batch_y_transformations = batch_y_transformations,
                meters = train_meters)
        valid_avg_metric, valid_avg_criterion = validate(
                valid_loader, model,
                criterion, metric,
                train_epoch.current_epoch,
                print_fn = print_fn,
                batch_x_transformations = batch_x_transformations,
                batch_y_transformations = batch_y_transformations,
                meters = valid_meters)
        if train_epoch.best_attained_metric < valid_avg_metric:
            print_fn(f"Saving snapshot after attaining results over validation data {valid_avg_metric}")
            train_epoch.best_attained_metric = valid_avg_metric
            state = {
                'model': model.state_dict(),
                'best_attained_metric': train_epoch.best_attained_metric}
            torch.save(state, snap_shot_location)
        train_epoch.current_epoch += 1
        return train_avg_metric, train_avg_criterion, valid_avg_metric, valid_avg_criterion, optimizer.param_groups[0]['lr']
    else:
        raise no_further_epochs('Reached the set epoch count')