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

+ To train on CIFAR-100 with IID data partition and noise setting $(\rho,\tau)=(0.6,0.5)$, over 50 clients:

```
python main.py --dataset cifar100 --model resnet34 --non_iid_prob_class 0.7 --alpha_dirichlet 10  --rounds 500 --seed 1 --mixup --lr 0.03 --beta 5 --ratio 0.4
```

+ Please find more details of training over iNaturalist dataset in FedGrab [code](https://github.com/ZackZikaiXiao/FedGraB)

