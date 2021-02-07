



with open('tuning.csv', 'w') as f:
    f.write(','.join(['run_id', 'epoch', 'loss_train', 'loss_test', 'acc_train', 'acc_test', 'num_layers', 'num_epochs', 'batch_size', 'hidden_size', 'learntime']) + '\n')
    for k in d.keys():
        for i in range(d[k]['epochs']):
            f.write("{run_id}, {epoch:d}, {loss_train:.8f},{loss_test:.8f},{acc_train:.8f},{acc_test:.8f},{num_layers:d},{num_epochs:d},{batch_size},{hidden_size:d},{learntime:.8f}\n".format(
                        run_id = k,
                        epoch = i,
                        loss_train = d[k]['loss'][i],
                        loss_test = d[k]['val_loss'][i],
                        acc_train = d[k]['binary_accuracy'][i],
                        acc_test = d[k]['val_binary_accuracy'][i],
                        num_layers = d[k]['layers'],
                        num_epochs = d[k]['epochs'],
                        batch_size = d[k]['batch_size'],
                        hidden_size = d[k]['hidden_size'],
                        learntime = d[k]['learntime']))