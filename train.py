from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import *


def train_epoch(model, loader, optimizer, dev, input_n, output_n, epoch, tb_writer, clip_grad=None):
    running_loss = 0
    n = 0
    device = dev
    for cnt, batch in tqdm(enumerate(loader), total=len(loader)):
        batch = batch.to(device)
        batch_dim = batch.shape[0]
        n += batch_dim

        sequences_train = batch[:, 0:input_n, :]
        sequences_gt = batch[:, input_n:input_n + output_n, :]
        optimizer.zero_grad()
        pred = model(sequences_train)
        loss = mpjpe_error(pred, sequences_gt)
        tb_writer.add_scalar('loss/train', loss.item(), cnt+epoch*len(loader))
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad)

        optimizer.step()
        running_loss += loss * batch_dim

    return running_loss.detach().cpu() / n


def train_epoch_autoreg(model, loader, optimizer, dev, input_n, output_n, epoch, tb_writer, clip_grad=None):
    running_loss = 0
    n = 0
    device = dev
    for cnt, batch in tqdm(enumerate(loader), total=len(loader)):
        batch = batch.to(device)
        batch_dim = batch.shape[0]
        n += batch_dim
        sequences_train = batch[:, 0:input_n, :]
        sequences_gt = batch[:, input_n:input_n + output_n, :]
        pred = torch.Tensor().to(device)
        for i in range(output_n):
            if i == 0:
                input_train = sequences_train
            elif i >= input_n:
                input_train = pred[:, (i-input_n):, :]
            else:
                input_train = sequences_train[:, i:, :]
                input_train = torch.cat((input_train, pred), 1)
            optimizer.zero_grad()
            pred_one = model(input_train)[:, 0, :].unsqueeze(1)
            # pred_one = model(input_train)
            pred = torch.cat((pred, pred_one), 1)
        loss = mpjpe_error(pred, sequences_gt)
        tb_writer.add_scalar('loss/train', loss.item(), cnt+epoch*len(loader))
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad)

        running_loss += loss * batch_dim
        optimizer.step()

    return running_loss.detach().cpu() / n


@torch.no_grad()
def eval_epoch(model, loader, dev, input_n, output_n, epoch, tb_writer):
    running_loss = 0
    n = 0
    device = dev
    for cnt, batch in enumerate(loader):
        batch = batch.to(device)
        batch_dim = batch.shape[0]
        n += batch_dim

        sequences_train = batch[:, 0:input_n, :]
        sequences_gt = batch[:, input_n:input_n + output_n, :]
        pred = model(sequences_train)

        loss = mpjpe_error(pred, sequences_gt)
        tb_writer.add_scalar('loss/val', loss.item(), cnt + epoch * len(loader))
        running_loss += loss * batch_dim

    return running_loss.detach().cpu() / n


@torch.no_grad()
def eval_epoch_autoreg(model, loader, dev, input_n, output_n, epoch, tb_writer):
    running_loss = 0
    n = 0
    device = dev
    for cnt, batch in enumerate(loader):
        batch = batch.to(device)
        batch_dim = batch.shape[0]
        n += batch_dim

        sequences_train = batch[:, 0:input_n, :]
        sequences_gt = batch[:, input_n:input_n + output_n, :]
        pred = torch.Tensor().to(device)
        for i in range(output_n):
            if i == 0:
                input_train = sequences_train
            elif i >= input_n:
                input_train = pred[:, (i-input_n):, :]
            else:
                input_train = sequences_train[:, i:, :]
                input_train = torch.cat((input_train, pred), 1)
            # pred_one = model(input_train)
            pred_one = model(input_train)[:, 0, :].unsqueeze(1)
            pred = torch.cat((pred, pred_one), 1)
        loss = mpjpe_error(pred, sequences_gt)
        tb_writer.add_scalar('loss/val', loss.item(), cnt + epoch * len(loader))
        running_loss += loss * batch_dim

    return running_loss.detach().cpu() / n


def train(model, train_loader, valid_loader, device, autoreg,
          lr=0.001, n_epochs=30, root='./runs', input_n=10, output_n=10):
    log_dir = get_log_dir(root)
    tb_writer = SummaryWriter(log_dir=log_dir)
    print('Save data of the run in: %s' % log_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.3, verbose=True)

    train_loss, val_loss, test_loss = [], [], []

    for epoch in range(n_epochs):
        print('Run epoch: %i' % epoch)

        model.train()
        if autoreg:
            running_loss = train_epoch_autoreg(model, train_loader, optimizer, device,
                                               input_n, output_n, epoch, tb_writer)
        else:
            running_loss = train_epoch(model, train_loader, optimizer, device, input_n, output_n, epoch, tb_writer)
        train_loss.append(running_loss)

        model.eval()

        if autoreg:
            running_loss = eval_epoch_autoreg(model, valid_loader, device, input_n, output_n, epoch, tb_writer)
        else:
            running_loss = eval_epoch(model, valid_loader, device, input_n, output_n, epoch, tb_writer)

        # running_loss = eval_epoch(model, valid_loader, device, input_n, output_n, epoch, tb_writer)
        val_loss.append(running_loss)

        scheduler.step(val_loss[-1].item())

        tb_writer.add_scalar('loss_epoch/train', train_loss[-1].item(), epoch)
        tb_writer.add_scalar('loss_epoch/val', val_loss[-1].item(), epoch)
        torch.save(model.state_dict(), os.path.join(log_dir, 'model.pt'))
