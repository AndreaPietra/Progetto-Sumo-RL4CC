# Progetto-Sumo-RL4CC

## Installation instructions

### Linux Ubuntu installation

1. Install python (version 3.10)

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.10
sudo apt-get install python3.10-venv
```

2. Install SUMO

```
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc

# Set SUMO_HOME environment variable
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```

3. Install `xvfb` (to render in rgb array format)

```
# 1. Confirm release and that this *is* noble
lsb_release -a
cat /etc/os-release

# 2. Confirm the package is visible to this APT instance
apt-cache policy xvfb
apt-cache search xvfb

# 3. Refresh cache and install
sudo apt update
sudo apt install xvfb
```

4. Create virtual environment

```
python3.10 -m venv .env
source .env/bin/activate
pip install --upgrade pip
```

5. Install python requirements

```
pip install -r requirements.txt
```
