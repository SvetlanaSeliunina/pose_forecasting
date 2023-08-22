from utils import *


@torch.no_grad()
def test(model, train_loader, test_loader, extra_loader, dev, input_n, output_n):
    train_loss, test_loss, extra_loss = 0, 0, 0
    train_seq, test_seq, extra_seq = [], [], []
    device = dev

    n = 0
    for cnt, batch in enumerate(train_loader):
        batch = batch.to(device)
        batch_dim = batch.shape[0]
        n += batch_dim

        sequences_train = batch[:, 0:input_n, :]
        sequences_gt = batch[:, input_n:input_n + output_n, :]
        sequences_predict = model(sequences_train)
        loss = mpjpe_error(sequences_predict, sequences_gt)
        train_loss += loss * batch_dim
        if cnt == len(train_loader)-1:
            train_input = batch[0, :, :].cpu()
            train_seq.append(train_input)
            train_output = torch.cat((sequences_train, sequences_predict), 1)[0, :, :].cpu()
            train_seq.append(train_output)
    train_loss = train_loss.detach().cpu() / n

    n = 0
    for cnt, batch in enumerate(test_loader):
        batch = batch.to(device)
        batch_dim = batch.shape[0]
        n += batch_dim

        sequences_train = batch[:, 0:input_n, :]
        sequences_gt = batch[:, input_n:input_n + output_n, :]
        sequences_predict = model(sequences_train)
        loss = mpjpe_error(sequences_predict, sequences_gt)
        test_loss += loss * batch_dim
        if cnt == len(test_loader)-1:
            test_input = batch[0, :, :].cpu()
            test_seq.append(test_input)
            test_output = torch.cat((sequences_train, sequences_predict), 1)[0, :, :].cpu()
            test_seq.append(test_output)
    test_loss = test_loss.detach().cpu() / n

    n = 0
    for cnt, batch in enumerate(extra_loader):
        batch = batch.to(device)
        batch_dim = batch.shape[0]
        n += batch_dim

        sequences_train = batch[:, 0:input_n, :]
        sequences_gt = batch[:, input_n:input_n + output_n, :]
        sequences_predict = model(sequences_train)
        loss = mpjpe_error(sequences_predict, sequences_gt)
        extra_loss += loss * batch_dim
        if cnt == len(extra_loader) - 1:
            extra_input = batch[0, :, :].cpu()
            extra_seq.append(extra_input)
            extra_output = torch.cat((sequences_train, sequences_predict), 1)[0, :, :].cpu()
            extra_seq.append(extra_output)
    extra_loss = extra_loss.detach().cpu() / n

    return train_loss.item(), test_loss.item(), extra_loss.item(), train_seq, test_seq, extra_seq
