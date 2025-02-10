# Recurrent segmentation

This is the github repository referent to the paper [Feedback Emerges in Noisy and Few-Shot Environments](https://en.wikipedia.org/wiki/HTTP_404). If you're having trouble running the code, please open an issue.


## Setup

Run the config.sh script to create an environment with all the packages that are necessary:

```
bash setup/config.sh <ENVNAME>
```

This script setups an anaconda environment with the following requirements:

| Python package | Version |
|:---------------|:-------:|
| **torch**		 | 1.13.0+cu117|
| **skimage** 	 | 0.23.2 |
| **cv2**		 | 4.9.0 |
| **numpy**		 | 1.24.3 |
| **torchvision**| 0.14.0+cu117 |
| **PIL**		 | 10.2.0 |
| **segmentation_models_pytorch** | 0.1.3 |

## Citation

If you found this repository useful and used it in your research, please do not forget to cite it as:
```
@article{calhas2025,
  title={},
  author={Calhas, David and Oliveira, Arlindo L},
  journal={arXiv preprint},
  year={2025}
}
```