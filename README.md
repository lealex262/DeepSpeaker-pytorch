# Deep Speaker from Baidu Research -  Pytorch Implementation

This is a slightly modified pytorch implementation of the model(modified Resnet + triplet loss) presented by Baidu Research in [Deep Speaker: an End-to-End Neural Speaker Embedding System](https://arxiv.org/pdf/1705.02304.pdf).

This code was tested using Voxceleb database. [Voxceleb database paper](https://www.robots.ox.ac.uk/~vgg/publications/2017/Nagrani17/nagrani17.pdf)


## Credits
Original paper:
- Baidu Research paper:
```
@article{DBLP:journals/corr/LiMJLZLCKZ17,
  author    = {Chao Li and Xiaokong Ma and Bing Jiang and Xiangang Li and Xuewei Zhang and Xiao Liu and Ying Cao and Ajay Kannan and Zhenyao Zhu},
  title     = {Deep Speaker: an End-to-End Neural Speaker Embedding System},
  journal   = {CoRR},
  volume    = {abs/1705.02304},
  year      = {2017},
  url       = {http://arxiv.org/abs/1705.02304},
  timestamp = {Wed, 07 Jun 2017 14:41:04 +0200},
  biburl    = {http://dblp.org/rec/bib/journals/corr/LiMJLZLCKZ17},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```

Also, use the part of code:
- [liorshk's git repository](https://github.com/liorshk/facenet_pytorch)
   - Baseline code - Facenet pytorch implimetation
- [hbredin's git repository](https://github.com/hbredin/pyannote-db-voxceleb)
   - Voxceleb Database reader
- [qqueing/s git repository](https://github.com/qqueing/DeepSpeaker-pytorch)


## How to Run
Train

'''bash
python train_triplet.py
'''

Preprocessing
'''bash
python train_triplet.py --makemfb
python train_triplet.py --makeif
python train_triplet.py --makemel
'''



Data Structure
```
dataroot
└── dev
    └── id #person
        └── url
            ├── .wav file
            ├── .npy file
└── Test
└── dev
```
