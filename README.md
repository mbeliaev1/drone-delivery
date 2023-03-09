# Congestion Aware Routing with Drones
This directory is supplementary material for our work published in MDPI Future Transportation: 

[Congestion-Aware Bi-Modal Delivery Systems Utilizing Drones](https://www.mdpi.com/2673-7590/3/1/20) Mark Beliaev, Negar Mehr, Ramtin Pedarsani.

All relevant citations for methods used are found in the paper's list of references.

## Requirements

We recommend using pacakge manager [pip](https://pip.pypa.io/en/stable/) as well as 
[cuda](https://developer.nvidia.com/cuda-toolkit) to install the relative packages:

**sumo:**
https://sumo.dlr.de/docs/Downloads.php#linux_binaries

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
export SUMO_HOME="/usr/share/sumo"
```

**conda:**

- python-3.9.1 [python](https://www.python.org/downloads/release/python-391/)
- numpy-1.19.2 [numpy](https://numpy.org/devdocs/release/1.19.2-notes.html)
- matplotlib-3.3.4 [matplotlib](https://matplotlib.org/3.3.4/)
- pandas-1.2.1 [pandas](insertlink)
- networkx-2.5 [networkx](insertlink)
- jupyter-1.0.0 [jupyter](insertlink)
- scipy-1.6.2 [scipy](insertlink)

**pip:**
- tqdm-4.56.2 [tqdm](insterlink)
- geopy-2.1.0 [geopy](insterlink)
- qpsolvers-1.5 [qpsolvers](https://pypi.org/project/qpsolvers/)
## Usage


