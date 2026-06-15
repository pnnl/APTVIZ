# Compositional Community Detection: Automated identification of chemical segregation in atom probe tomography data

Compositional Community Detection (CCD) is used to identify compositionally distinct regions in reconstructed APT point clouds. The CCD workflow preprocesses raw APT data into overlapping spherical neighborhoods, clusters them by composition via k-means, computes Kolmogorov-Smirnov statistics, and applies Louvain community detection to reveal dominant compositional domains.

## Modes of use

### Agent skill
An agent skill is located in the '.github/skills' folder. The skill was created for use with GitHub Copilot but can be used with other agentic systems by following that system's directory layout. The `SKILL.md` file  provides a detailed overview of the CCD process.

### Interactive GUI
A web-app GUI is available in the `apt_viz` folder.

### Stand-alone scripts
Stand-alone scripts can befound in `the scripts` folder



## Installation
#### Python Environment
Tested with Python 3.11

Packages:
* scipy (must be v1.10.1)
* numpy (tested on v1.26.4)
* pandas (tested on v1.5.3)
* scikit-learn (tested on v1.2.1)
* python-louvain (tested on v0.16)
* OPTICS-APT (https://github.com/pnnl/APT)


## Reference
If you use this in your work please cite:
```
@article{bilbrey2025compositional,
  title={Compositional Community Detection: Automated Identification of Chemical Segregation in Atom Probe Tomography Data},
  author={Bilbrey, Jenna A and Doty, Christina and Wirth, Mark G and Tong, Mengkong and Royer, Jacqueline and Senor, David J and Devaraj, Arun},
  journal={Microscopy and Microanalysis},
  volume={31},
  number={3},
  pages={ozaf036},
  year={2025},
  publisher={Oxford University Press US}
}
```
