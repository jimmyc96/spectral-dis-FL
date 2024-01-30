# Spectral Co-Distillation for Personalized Federated Learning

This is the code for "Spectral Co-Distillation for Personalized Federated Learning" accepted by NeurIPS 2023([Paper link](https://openreview.net/forum?id=RqjQL08UFc))



## 1. Parameters

**1.1. Descriptions**

| parameters           | description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `rounds`             | Number of total training rounds                              |
| `num_users`          | Number of clients                                            |
| `local_bs`           | Batch size for local training                                |
| `beta`               | Coefficient for local proximal term, default: `1`            |
| `ratio`              | Ratio for spectra truncation                                 |
| `model`              | neural network model                                         |
| `dataset`            | Dataset, options:`cifar10`,`cifar100`                        |
| `iid`                | `Action` IID or non-IID data partition, default: `store_true` |
| `non_iid_prob_class` | Non-IID sampling probability for class                       |
| `alpha_dirichlet`    | Parameter for Dirichlet distribution ($\alpha_{DIR}$ in the paper) |
| `pretrained`         | `Action`, whether to use pre-trained model, default: `store_true` |
| `mixup`              | `Action`, whether to use Mixup, default: `store_true`        |
| `alpha`              | Parameters for Mixup, default: `1`                           |



## 2. How to use the code

+ To train on CIFAR-10 with non-IID data partition with $(p,\alpha_{Dir})=(0.7,10)$  over 100 clients:

```
python main.py --dataset cifar10 --model resnet18 --non_iid_prob_class 0.7 --alpha_dirichlet 10  --rounds 500 --seed 1 --mixup --lr 0.03 --beta 5 --ratio 0.4
```

+ To train on CIFAR-100 with non-IID data partition, over 50 clients:

```
python main.py --dataset cifar100 --model resnet34 --non_iid_prob_class 0.7 --alpha_dirichlet 10  --rounds 500 --seed 1 --mixup --lr 0.03 --beta 5 --ratio 0.4
```

+ Please find more details of training over iNaturalist dataset in FedGrab [code](https://github.com/ZackZikaiXiao/FedGraB)

#### Communication efficiency evaluation

The overall communication efficiency speedup could be obtained by time tracking. You can use any package which can achieve this objective.  While various packages are available for this purpose, a straightforward method is to use Python’s built-in `time` module. Begin by importing this module and initiating a timer just before the start of the federated training process, as shown: `start_time = time.time()`. After the completion of a training round on each node, log the end time with `end_time = time.time()`. The difference between these two, calculated as `total_training_time = end_time - start_time`, represents the total duration of the training process. To specifically measure the communication time, record timestamps right before and after each phase of data transmission. Accumulating these time segments will give you the total communication time. The efficiency speedup is then derived by comparing this total communication time against the overall training time. This comparison yields a tangible measure of the efficiency enhancements in our federated learning model. It’s important to note that communication speedup may vary based on the training environment and hardware configurations.
