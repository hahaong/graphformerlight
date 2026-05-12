

# GraphFormerLight: A Topology-Aware Multi-Agent Reinforcement Learning Architecture for Proactive Traffic Signal Control

This repository contains the official implementation of **GraphFormerLight**, an advanced Multi-Agent Reinforcement Learning (MARL) framework designed for large-scale Adaptive Traffic Signal Control (ATSC). 

By integrating a Sequence-to-Sequence (Seq2Seq) traffic prediction module with a centralized Graph Mixing Network (GMN), GraphFormerLight proactively mitigates traffic congestion while solving the over-smoothing and credit assignment problems inherent in deep MARL.

## 📢 Acknowledgments
This codebase is proudly built upon and extends two highly influential open-source projects:
* **[PyMARL](https://github.com/oxwhirl/pymarl):** The underlying MARL framework and centralized training with decentralized execution (CTDE) architecture are based on PyMARL.
* **[SUMO-RL](https://github.com/LucasAlegre/sumo-rl):** The traffic simulation environment and intersection agent wrappers are built upon SUMO-RL, interfacing with the Eclipse SUMO (Simulation of Urban MObility) microscopic traffic simulator.

## ✨ Key Features
* **Proactive Traffic Prediction:** Integrates a Seq2Seq transformer module to anticipate near-future traffic arrivals, shifting agents from reactive to proactive control.
* **Topology-Aware Graph Mixing Network:** Utilizes a single-layer Graph Attention Network (GAT) restricted by the physical road adjacency matrix to prevent feature over-smoothing while efficiently assigning multi-agent credit.
* **Reward Fraction Mechanism:** Dynamically balances global network throughput with local intersection efficiency using contextualized graph embeddings.
* **Real-World Benchmarks:** Includes heterogeneous, real-world traffic network topologies (Hangzhou, Cologne, Jinan, New York) alongside synthetic grid maps.

Here is the comprehensive installation guide tailored for your GitHub `README.md`.

Because your project relies heavily on **PyTorch**, **PyTorch Geometric (PyG)**, and specific CUDA versions, the best practice is to provide users with an `environment.yml` (for Conda) and a `requirements.txt` (for pip), while giving explicit instructions on how to handle the GPU-accelerated libraries.

You can copy and paste this directly into your `README.md` file, replacing the previous installation section.

---

## 🛠️ Installation & Requirements

### 1. System Requirements (Eclipse SUMO)
GraphFormerLight requires the microscopic traffic simulator **Eclipse SUMO** (Simulation of Urban MObility) to run the environments.
* **Download:** [SUMO Downloads](https://sumo.dlr.de/docs/Downloads.php) (Recommended version: 1.18.0+)
* **Environment Variable:** You must set the `SUMO_HOME` environment variable to the root directory of your SUMO installation.
  * *Linux/Mac:* `export SUMO_HOME="/usr/share/sumo"`
  * *Windows:* `setx SUMO_HOME "C:\Program Files (x86)\Eclipse\Sumo"`

### 2. Clone the Repository
```bash
git clone [https://github.com/yourusername/GraphFormerLight.git](https://github.com/yourusername/GraphFormerLight.git)
cd GraphFormerLight
```

### 3. Python Environment Setup

We recommend using **Conda** for isolated environments, especially to handle CUDA, PyTorch, and PyTorch Geometric (PyG) dependencies seamlessly. The code is tested on **Python 3.11**.

#### Option A: Using Conda (Recommended)

You can recreate the exact environment using the provided `environment.yml` file.

1. Create the `environment.yml` file in the root of your project:
```yaml
name: sumorl
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy=2.0.1
  - pandas=2.3.2
  - matplotlib=3.10.5
  - seaborn=0.13.2
  - gymnasium=0.28.1
  - pyyaml=6.0.2
  - h5py=3.15.1
  - pip:
    - pettingzoo==1.24.3
    - sumolib==1.18.0
    - traci==1.18.0
    - libsumo==1.18.0
    - networkx==3.5
    - scipy==1.16.1
    - tqdm==4.67.1

```


2. Create and activate the environment:
```bash
conda env create -f environment.yml
conda activate sumorl

```


3. Install PyTorch for your specific CUDA version (the default setup uses CUDA 12.1/12.8). Adjust the index URL if you do not have a GPU:
```bash
# Install PyTorch
pip install torch==2.8.0 torchvision==0.23.0 --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
```



#### Option B: Using Pip (`requirements.txt`)

If you prefer standard pip, create a virtual environment and install the dependencies from `requirements.txt`.

1. Create a `requirements.txt` file in your root directory:
```text
numpy==2.0.1
pandas==2.3.2
matplotlib==3.10.5
seaborn==0.13.2
gymnasium==0.28.1
pettingzoo==1.24.3
pyyaml==6.0.2
h5py==3.15.1
sumolib==1.18.0
traci==1.18.0
libsumo==1.18.0
networkx==3.5
scipy==1.16.1
tqdm==4.67.1

```


2. Install the requirements:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

```


3. Install the Pytorch:
```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)

```

Here is a professionally formatted "Usage" section that you can copy and paste directly into your `README.md`. It clearly presents the command and breaks down the purpose of each argument for users.

---

## 🚀 Running GraphFormerLight

To start training the GraphFormerLight model, you will use the `main.py` script. The command requires several specific arguments to define the environment files, output files, and algorithmic modules.

Below is the standard command to run the model on the Hangzhou (6985 veh/hr) dataset:

```bash
python main.py \
  --csv_name graphmix_numrun1_hangzhou16_6985 \
  --config graphmix \
  --adj_mask_file maps/hangzhou16/anon_4_4_hangzhou_real.csv \
  --seq2seq \
  --on_policy_learning \
  --net_file maps/hangzhou16/anon_4_4_hangzhou_real_6985.net.xml \
  --route_file maps/hangzhou16/anon_4_4_hangzhou_real_6985.rou.xml
```

### 📋 Command Line Arguments Explained

| Parameter | Description |
| --- | --- |
| `--csv_name` | The name of the output `.csv` file where training results will be stored (includes reward, accumulated waiting time, mean speed, total stopped vehicles, and attention scores). |
| `--config` | Specifies the algorithm configuration to use (e.g., `graphmix` for GraphFormerLight). |
| `--adj_mask_file` | **Mandatory.** The path to the adjacency matrix file. This provides the physical topology information required by the Graph Mixing Network. |
| `--seq2seq` | A flag that activates the Sequence-to-Sequence (Seq2Seq) traffic prediction module. |
| `--on_policy_learning` | A flag that sets the reinforcement learning scheme to on-policy. |
| `--net_file` | The path to the SUMO network (`.net.xml`) file defining the road infrastructure. |
| `--route_file` | The path to the SUMO route (`.rou.xml`) file defining the vehicle traffic flow. |

**Note:** If you want to test the model on different datasets (like Cologne or Jinan), simply update the paths for the `--adj_mask_file`, `--net_file`, and `--route_file` to point to the respective map directories.
