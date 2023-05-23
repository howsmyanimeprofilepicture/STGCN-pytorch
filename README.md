# STGCN Pytorch

* This repository is the Pytorch implmentation of the [STGCN](https://arxiv.org/abs/1709.04875) that is designed for a graph timeseries forecasting.
* In fact, all codes in this repo are nothing but a clone code of the Keras tutorial — Arash Khodadadi(2021) - ["Traffic forecasting using graph neural networks and LSTM"](https://keras.io/examples/timeseries/timeseries_traffic_forecasting/).
* The only difference is that I coded it using the Pytorch and I modified some train options...

```bash
# DOWNLOAD_DATASET
wget https://github.com/VeritasYin/STGCN_IJCAI-18/raw/master/dataset/PeMSD7_Full.zip
````

* The dataset used in this repo is the ["PeMSD7"](https://pems.dot.ca.gov/) which was collected from Caltrans Performance Measurement System (PeMS) in real-time by over 39, 000 sensor stations, deployed across the major metropolitan areas of California state highway system.
* If you want to know a theoretical concept, I recommend this [paper](https://arxiv.org/abs/1709.04875) and [GitHub repository](https://github.com/VeritasYin/STGCN_IJCAI-18).
