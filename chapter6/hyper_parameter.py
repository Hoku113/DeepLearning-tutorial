import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from chapter3.dataset.mnist import load_mnist
from shared_code.multi_layer_net import MultiLayerNet
from shared_code.util import shuffle_dataseet
from shared_code.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# delete training data for rapidry learning
x_train = x_train[:500]
t_train = t_train[:500]

# split validation data
validation_rate = 0.2
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataseet(x_train, t_train)
x_val = x_train[:validation_num]
t_val = x_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]

def __train(lr, weight_decay, epochs=50):
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                             output_size=10, weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epochs, mini_batch_size=100,
                      optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)

    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list

# Random search for hyper parameter
optimization_trial = 100
results_val = {}
results_train = {}

for _ in range(optimization_trial):
    # Specifying the range which is searched hyper parameter
    weight_decay = 10 ** np.random.uniform(-8, -4)
    lr = 10 ** np.random.uniform(-6, -2)

    val_acc_list, train_acc_list = __train(lr, weight_decay)
    print(f"val acc: {val_acc_list[-1]} | lr: {lr} , weight decay: {weight_decay}")
    key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

# drao graph
print("========= Hyper-Parameter Optimization Result =============")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

for key, val_acc_list in sorted(results_val.items(), key=lambda x: x[1][-1], reverse=True):
    print(f"Best- {i+1} (val acc: {val_acc_list[-1]} | {key}")

    plt.subplot(row_num, col_num, i+1)
    plt.title(f"Best-{i+1}")
    plt.ylim(0.0, 1.0)
    if i % 5: plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    i += 1

    if i >= graph_draw_num:
        break

plt.show()

