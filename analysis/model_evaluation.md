# Baseline:
- predicting every picture is the same whale
- score: 34.48176


# Lenet Model:
- whale_default_lenet
- kaggle: 32.61463
- train logloss: 33.86
- accuracy: 0.28917253521126762

# crazy Lenet Model:
- model name: kaggle_whale_lenet_5000
- epochs: 5000; solver_type: nesterov_accelerated gradient ; base_learning_rate: 0.005, policy: exponential_delay; gamma: 0.95
- accuracy: 0.53345070422535212
- logloss: 34.384051600537781
- kaggle: 32.46881


# googlenet Model:
- model name: kaggle_whale_googlenet
- accuracy: 0.009683098591549295
- logloss: 31.385220036173479
- kaggle: 32.08149
