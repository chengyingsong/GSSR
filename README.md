# GSSR

**This is an implementation of ["Group Shuffle and Spectral-Spatial Fusion for Hyperspectral Image Super-Resolution"](https://ieeexplore.ieee.org/abstract/document/10011536).**

Dataset
------
**Three public datasets, i.e., [CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/ "CAVE"), [Chikusei](https://naotoyokoya.com/Download.html)), are employed to verify the effectiveness of the  proposed GSSR. Since there are too few images in these datasets for deep learning algorithm, we augment the training data. With respect to the specific details, please see the implementation details section.**

**Moreover, we also provide the code about data pre-processing in folder. The folder contains three parts, including training set augment, test set pre-processing, and band mean for all training set.**


Train and Test
--------
**The ADAM optimizer with beta_1 = 0.9, beta _2 = 0.999 is employed to train our network.  The learning rate is initialized as 10^-4 for all layers, which decreases by a half at every 35 epochs.**

**You can train or test directly from the command line as such:**

###### # python train.py --cuda --gpus 0 --datasetName CAVE  --upscale_factor 4
###### # python test.py --cuda --model_name checkpoint/*.pth

Citation 
--------
**Please consider cite this paper if you find it helpful.**
  
@ARTICLE{10011536,
		author={Wang, Xinya and Cheng, Yingsong and Mei, Xiaoguang and Jiang, Junjun and Ma, Jiayi},
		journal={IEEE Transactions on Computational Imaging}, 
		title={Group Shuffle and Spectral-Spatial Fusion for Hyperspectral Image Super-Resolution}, 
		year={2022},
		volume={8},
		number={},
		pages={1223-1236},
		doi={10.1109/TCI.2023.3235153}
}

--------
If you has any questions, please send e-mail to @gmail.com.

