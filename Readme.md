# Variational Hierarchical Conversation RNN (VHCR)
[PyTorch 0.4](https://github.com/pytorch/pytorch) Implementation of ["Towards an Automatic Turing Test, Learning to Evaluate Dialogue Responses"](https://arxiv.org/abs/1708.07149) accepted in ACL 2017.

Our ADEM model is baed on VHRED model in ["A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling" repo."](https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling).

## Training


To run training:
```
python train.py --data=<data> --model=<model> --batch_size=<batch_size>
```

For example:

```
CUDA_VISIBLE_DEVICES=3 python train.py --mode train --temperature 1.0 --beam_size 5 --model ADEM --context_size 400 --embedding_size 300 --encoder_hidden_size 400 --bidirectional True --decoder_hidden_size 800  --data smart_ko_adem --weight_decay 0.075 --pretrained_path pretrained_vhred.pkl --n_epoch 50
```


By default, it will save a model checkpoint every epoch to <save_dir> and a tensorboard summary.
For more arguments and options, see config.py.


## Evaluation
To evaluate the word perplexity:
```
python eval.py --model=<model> --checkpoint=<path_to_your_checkpoint>
```
For example:
```
CUDA_VISIBLE_DEVICES=3 python eval.py --model ADEM  --context_size 400 --embedding_size 300 --encoder_hidden_size 400 --bidirectional True --data smart_ko_adem --checkpoint smart_ko_adem/ADEM/2018-07-25_143225/50.pkl --test_res_path adem_sigmoid_ep50_test.txt --test_raw_score_path adem_sigmoid_ep50_raw_score.txt
```
