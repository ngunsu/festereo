# Fast CNN Stereo Depth Estimation through Embedded GPU Device

This is the implementation of our paper"[Fast CNN Stereo Depth Estimation through Embedded GPU Device](https://www.mdpi.com/1424-8220/20/11/3249)"

### Results

| Model | Dataset | EPE | Err > 3 |
|:-----:|:-------:|:---:|:-------:|
| Default| Kitti2012| 1.80 | 0.11 |  


### Reproduce results 

##### Docker 

```bash
./docker/launch.sh

# Sceneflow training
python cli.py festereo-train --num_workers 16 --max_epochs 20 --min_epochs 1 --patience 100 --lr 5e-3 --save_top_k 20

# Kitti2012 (using sceneflow pretrained)
python cli.py festereo-train --num_workers 16 --max_epochs 300 --min_epochs 200 --patience 100 --lr 5e-3 --dataset kitti2012 --pretrained [path]/sceneflow_ckpt_epoch_19.ckpt --scheduler plateau 
```

### Pretrained networks

- [Sceneflow checkpoint epoch 19](https://www.dropbox.com/s/a3ry4lqouw7nkhc/sceneflow.ckpt?dl=0)
- [Kitti2012](https://www.dropbox.com/s/ckdixxrp7kb67b4/kitti2012.ckpt?dl=0)

### TODO

- Improve documentation 
- Add conda support
- Add fast inference
- Add Jetson instructions

#### Citation
```
@article{Aguilera_2020,
	doi = {10.3390/s20113249},
	url = {https://doi.org/10.3390%2Fs20113249},
	year = 2020,
	month = {jun},
	publisher = {{MDPI} {AG}},
	volume = {20},
	number = {11},
	pages = {3249},
	author = {Cristhian A. Aguilera and Cristhian Aguilera and Crist{\'{o}}bal A. Navarro and Angel D. Sappa},
	title = {Fast {CNN} Stereo Depth Estimation through Embedded {GPU} Devices},
	journal = {Sensors}
} 
```
