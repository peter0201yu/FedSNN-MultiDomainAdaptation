# Multi-Domain Adaptation in a Federated Learning scenario with Spiking Neural Networks

## Acknowledgement
The federated learning framework and SNN implementation are based on this research: [Federated Learning with Spiking Neural Networks](https://github.com/Intelligent-Computing-Lab-Yale/FedSNN).

The multi-domain adaptation method is based on this research: [SFDA - Domain Adaptation without Source Data](https://github.com/youngryan1993/SFDA-SourceFreeDA).


## Environment

See `environment.yml`.


## Model

Four models in the `models.py`: VGG5 (ANN), VGG9 (ANN), VGG5 with BNTT (SNN), VGG9 with BNTT (SNN). The two ANNs use different feature extractors and share the same classifier (FC layers) given by `class CLS_ANN`, while the two SNNs are implemented separately.


## Datasets

The Digit5 dataset has 10 classes (digit 0-9) and 5 domains (mnist, mnist_m, synthetic, street-view-house-number, usps). You can download the binary data [here](https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/tree/master/M3SDA/code_MSDA_digit#digit-five-download).

The Office31 dataset has 31 classes (daily objects) and 3 domains (amazon, dslr, webcam). You can download the zipped images [here](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA).

To extract and preprocess the data for loading into PyTorch dataloader, I recommend using the [Dassyl](https://github.com/KaiyangZhou/Dassl.pytorch) repo. Follow the instructions in the [doc](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md) and you will get organized extracted images. For Digit5, I changed the script to divide the data by classes.


## Pretrain source domain model

VGG5 models (ANN and SNN) are enough for the Digit5 dataset, with the mnist domain being the source domain. To train the VGG5 ANN model, run `python pretrain_digit5.py`. To train the VGG5 SNN model, run `python pretrain_digit5.py --snn`. 

VGG9 models are used for the Office31 dataset, with the amazon domain being the source domain. To train the VGG9 ANN model, run `python pretrain_office31.py`. To train the VGG9 SNN model, run `python pretrain_office31.py --snn`. 

The two scripts are essentially the same, I separated them for convenience of loading/saving data and tuning hyperparameters. 

**Remember to change the hardcoded paths inside the scripts.**


## Single domain adaptation

`simplified_single.py` performs domain adaptation from the source domain (mnist) to one specified target domain. For example, to adapt the SNN model from mnist to usps, run `python simplified_single.py --snn --target usps`.


## Federated domain adaptation

`fed_mda_simplified.py` performs federated multi-domain adaptation. Provide arguments (explained in `./utils/options.py`) to customize the runs, fed-mda related arguments include:

- `num_users_per_domain`: number of users in each target domain, the data of each domain will be divided among those users according to the iid or non-iid sampling. 

- `mda_threshold_frac`: fraction of local data that will be chosen as confident training data after forwarding all local through the global model and ranking their entropies. For example, if `mda_threshold_frac = 0.2`, then for each client, the 20% of local data with least entropy will be used for local training.

_Note: I haven't implemented a way to automatically increase this fraction after certain number of rounds, but that can easily be done with code._

- `mda_threshold`: if you do not want to specify a fraction using `mda_threshold_frac`, you can specify a fixed number as the entropy threshold. If a data sample has an entropy smaller than this value, the sample and its predicted pseudo-label will be used for local training.

_Note: do not use `mda_threshold_frac` and `mda_threshold` simultaneously, they will conflict and an exception will be thrown._

Run `test_fed_mda.sh` as an example test script.
