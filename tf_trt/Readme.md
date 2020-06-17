

```
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

# Install NVIDIA driver
sudo apt-get install --no-install-recommends nvidia-driver-418
# Reboot. Check that GPUs are visible using the command: nvidia-smi
sudo apt-get update

sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.1-trt6.0.1.5-ga-20190913_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-cuda10.1-trt6.0.1.5-ga-20190913/7fa2af80.pub
sudo apt-get install -y --no-install-recommends --allow-downgrades libnvinfer6=6.0.1-1+cuda10.1 libnvinfer-dev=6.0.1-1+cuda10.1
```

ImageNet V1 224 partial log

```
2020-06-14 22:47:16.249947: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: tf_graph
2020-06-14 22:47:16.250013: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 235 nodes (-629), 234 edges (-656), time = 129.565ms.
2020-06-14 22:47:16.250030: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 401 nodes (166), 400 edges (166), time = 53ms.
2020-06-14 22:47:16.250053: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 347 nodes (-54), 346 edges (-54), time = 57.682ms.
2020-06-14 22:47:16.250073: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 122 nodes (-225), 121 edges (-225), time = 21171.2148ms.
2020-06-14 22:47:16.250090: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 122 nodes (0), 121 edges (0), time = 4.714ms.
2020-06-14 22:47:16.250107: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/TRTEngineOp_8_native_segment
2020-06-14 22:47:16.250124: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.779ms.
2020-06-14 22:47:16.250142: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 0.475ms.
2020-06-14 22:47:16.250159: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.73ms.
2020-06-14 22:47:16.250174: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.116ms.
2020-06-14 22:47:16.250185: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.72ms.
2020-06-14 22:47:16.250200: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/TRTEngineOp_6_native_segment
2020-06-14 22:47:16.250217: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.593ms.
2020-06-14 22:47:16.250235: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 0.423ms.
2020-06-14 22:47:16.250251: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.67ms.
2020-06-14 22:47:16.250266: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.085ms.
2020-06-14 22:47:16.250280: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.655ms.
2020-06-14 22:47:16.250296: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/TRTEngineOp_16_native_segment
2020-06-14 22:47:16.250312: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.535ms.
2020-06-14 22:47:16.250327: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 0.397ms.
2020-06-14 22:47:16.250340: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.64ms.
2020-06-14 22:47:16.250356: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.07ms.
2020-06-14 22:47:16.250372: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.644ms.
2020-06-14 22:47:16.250388: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/Logits/TRTEngineOp_0_native_segment
2020-06-14 22:47:16.250401: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 12 nodes (0), 11 edges (0), time = 4.559ms.
2020-06-14 22:47:16.250419: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 12 nodes (0), 11 edges (0), time = 3.932ms.
2020-06-14 22:47:16.250435: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 12 nodes (0), 11 edges (0), time = 3.735ms.
2020-06-14 22:47:16.250456: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 12 nodes (0), 11 edges (0), time = 0.519ms.
2020-06-14 22:47:16.250471: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 12 nodes (0), 11 edges (0), time = 3.747ms.
2020-06-14 22:47:16.250487: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/TRTEngineOp_19_native_segment
2020-06-14 22:47:16.250504: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.971ms.
2020-06-14 22:47:16.250518: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 0.675ms.
2020-06-14 22:47:16.250533: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.884ms.
2020-06-14 22:47:16.250549: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.123ms.
2020-06-14 22:47:16.250565: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.886ms.
2020-06-14 22:47:16.250579: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/TRTEngineOp_18_native_segment
2020-06-14 22:47:16.250593: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.655ms.
2020-06-14 22:47:16.250609: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 0.436ms.
2020-06-14 22:47:16.250626: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.687ms.
2020-06-14 22:47:16.250640: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.098ms.
2020-06-14 22:47:16.250654: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.626ms.
2020-06-14 22:47:16.250670: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/TRTEngineOp_7_native_segment
2020-06-14 22:47:16.250688: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 2.588ms.
2020-06-14 22:47:16.250702: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 2.025ms.
2020-06-14 22:47:16.250715: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 2.095ms.
2020-06-14 22:47:16.250731: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.297ms.
2020-06-14 22:47:16.250749: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 2.015ms.
2020-06-14 22:47:16.250763: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/TRTEngineOp_3_native_segment
2020-06-14 22:47:16.250776: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 1.552ms.
2020-06-14 22:47:16.250793: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 1.335ms.
2020-06-14 22:47:16.250810: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 1.346ms.
2020-06-14 22:47:16.250826: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.208ms.
2020-06-14 22:47:16.250839: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 1.443ms.
2020-06-14 22:47:16.250856: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/TRTEngineOp_22_native_segment
2020-06-14 22:47:16.250873: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.665ms.
2020-06-14 22:47:16.250887: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 0.442ms.
2020-06-14 22:47:16.250901: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.629ms.
2020-06-14 22:47:16.250917: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.086ms.
2020-06-14 22:47:16.250934: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.616ms.
2020-06-14 22:47:16.250948: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/TRTEngineOp_10_native_segment
2020-06-14 22:47:16.250962: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.565ms.
2020-06-14 22:47:16.250979: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 0.4ms.
2020-06-14 22:47:16.250996: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.648ms.
2020-06-14 22:47:16.251010: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.074ms.
2020-06-14 22:47:16.251023: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.615ms.
2020-06-14 22:47:16.251039: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/TRTEngineOp_4_native_segment
2020-06-14 22:47:16.251056: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.559ms.
2020-06-14 22:47:16.251070: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 0.415ms.
2020-06-14 22:47:16.251084: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.715ms.
2020-06-14 22:47:16.251100: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.076ms.
2020-06-14 22:47:16.251116: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.62ms.
2020-06-14 22:47:16.251131: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/TRTEngineOp_21_native_segment
2020-06-14 22:47:16.251145: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 1.017ms.
2020-06-14 22:47:16.251161: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 0.784ms.
2020-06-14 22:47:16.251177: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 1.047ms.
2020-06-14 22:47:16.251192: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.122ms.
2020-06-14 22:47:16.251206: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.979ms.
2020-06-14 22:47:16.251223: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/TRTEngineOp_26_native_segment
2020-06-14 22:47:16.251240: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.672ms.
2020-06-14 22:47:16.251254: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 0.437ms.
2020-06-14 22:47:16.251267: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.648ms.
2020-06-14 22:47:16.251283: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.083ms.
2020-06-14 22:47:16.251301: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.64ms.
2020-06-14 22:47:16.251315: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/TRTEngineOp_20_native_segment
2020-06-14 22:47:16.251328: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.609ms.
2020-06-14 22:47:16.251345: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 0.424ms.
2020-06-14 22:47:16.251362: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.621ms.
2020-06-14 22:47:16.251376: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.088ms.
2020-06-14 22:47:16.251389: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.643ms.
2020-06-14 22:47:16.251405: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/TRTEngineOp_2_native_segment
2020-06-14 22:47:16.251422: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.64ms.
2020-06-14 22:47:16.251437: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 0.445ms.
2020-06-14 22:47:16.251450: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.684ms.
2020-06-14 22:47:16.251467: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.093ms.
2020-06-14 22:47:16.251484: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.688ms.
2020-06-14 22:47:16.251499: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/TRTEngineOp_13_native_segment
2020-06-14 22:47:16.251512: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.633ms.
2020-06-14 22:47:16.251529: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 0.453ms.
2020-06-14 22:47:16.251546: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.722ms.
2020-06-14 22:47:16.251560: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.092ms.
2020-06-14 22:47:16.251574: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.716ms.
2020-06-14 22:47:16.251590: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/TRTEngineOp_11_native_segment
2020-06-14 22:47:16.251608: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.62ms.
2020-06-14 22:47:16.251622: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 0.447ms.
2020-06-14 22:47:16.251636: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.702ms.
2020-06-14 22:47:16.251653: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.083ms.
2020-06-14 22:47:16.251670: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.701ms.
2020-06-14 22:47:16.251684: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/TRTEngineOp_15_native_segment
2020-06-14 22:47:16.251697: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.663ms.
2020-06-14 22:47:16.251713: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 0.523ms.
2020-06-14 22:47:16.251731: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.674ms.
2020-06-14 22:47:16.251746: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.093ms.
2020-06-14 22:47:16.251760: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.738ms.
2020-06-14 22:47:16.251776: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/TRTEngineOp_5_native_segment
2020-06-14 22:47:16.251794: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 1.518ms.
2020-06-14 22:47:16.251808: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 1.389ms.
2020-06-14 22:47:16.251821: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 1.554ms.
2020-06-14 22:47:16.251837: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.191ms.
2020-06-14 22:47:16.251854: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 1.4ms.
2020-06-14 22:47:16.251869: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/TRTEngineOp_12_native_segment
2020-06-14 22:47:16.251882: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.58ms.
2020-06-14 22:47:16.251898: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 0.415ms.
2020-06-14 22:47:16.251916: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.651ms.
2020-06-14 22:47:16.251930: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.083ms.
2020-06-14 22:47:16.251943: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.659ms.
2020-06-14 22:47:16.251959: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/TRTEngineOp_23_native_segment
2020-06-14 22:47:16.251975: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 1.347ms.
2020-06-14 22:47:16.251989: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 1.336ms.
2020-06-14 22:47:16.252003: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 1.261ms.
2020-06-14 22:47:16.252019: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.17ms.
2020-06-14 22:47:16.252035: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 1.445ms.
2020-06-14 22:47:16.252050: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/TRTEngineOp_24_native_segment
2020-06-14 22:47:16.252065: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.608ms.
2020-06-14 22:47:16.252081: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 0.43ms.
2020-06-14 22:47:16.252093: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.749ms.
2020-06-14 22:47:16.252111: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.078ms.
2020-06-14 22:47:16.252128: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.654ms.
2020-06-14 22:47:16.252150: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_0/TRTEngineOp_1_native_segment
2020-06-14 22:47:16.252164: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.585ms.
2020-06-14 22:47:16.252178: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 0.406ms.
2020-06-14 22:47:16.252194: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.702ms.
2020-06-14 22:47:16.252212: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.071ms.
2020-06-14 22:47:16.252226: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.629ms.
2020-06-14 22:47:16.252239: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/TRTEngineOp_9_native_segment
2020-06-14 22:47:16.252255: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 4.091ms.
2020-06-14 22:47:16.252272: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 3.934ms.
2020-06-14 22:47:16.252286: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 3.721ms.
2020-06-14 22:47:16.252299: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.508ms.
2020-06-14 22:47:16.252316: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 3.415ms.
2020-06-14 22:47:16.252332: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/TRTEngineOp_17_native_segment
2020-06-14 22:47:16.252346: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.805ms.
2020-06-14 22:47:16.252360: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 0.567ms.
2020-06-14 22:47:16.252376: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.827ms.
2020-06-14 22:47:16.252392: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.107ms.
2020-06-14 22:47:16.252406: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.716ms.
2020-06-14 22:47:16.252419: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/TRTEngineOp_14_native_segment
2020-06-14 22:47:16.252436: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.56ms.
2020-06-14 22:47:16.252452: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 0.44ms.
2020-06-14 22:47:16.252467: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.649ms.
2020-06-14 22:47:16.252480: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.081ms.
2020-06-14 22:47:16.252496: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 0.662ms.
2020-06-14 22:47:16.252512: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/TRTEngineOp_27_native_segment
2020-06-14 22:47:16.252526: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 1.569ms.
2020-06-14 22:47:16.252540: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 1.437ms.
2020-06-14 22:47:16.252556: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 1.39ms.
2020-06-14 22:47:16.252572: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.219ms.
2020-06-14 22:47:16.252587: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 1.468ms.
2020-06-14 22:47:16.252600: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/TRTEngineOp_25_native_segment
2020-06-14 22:47:16.252616: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 1.405ms.
2020-06-14 22:47:16.252632: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 11 nodes (0), 10 edges (0), time = 1.432ms.
2020-06-14 22:47:16.252647: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 1.342ms.
2020-06-14 22:47:16.252660: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 11 nodes (0), 10 edges (0), time = 0.209ms.
2020-06-14 22:47:16.252676: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 11 nodes (0), 10 edges (0), time = 1.453ms.
```

ResNet-50  partial log

```
2020-06-14 22:48:50.399752: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: tf_graph
2020-06-14 22:48:50.399823: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 483 nodes (-272), 498 edges (-272), time = 408.338ms.
2020-06-14 22:48:50.399833: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 489 nodes (6), 504 edges (6), time = 284.582ms.
2020-06-14 22:48:50.399850: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 485 nodes (-4), 500 edges (-4), time = 248.565ms.
2020-06-14 22:48:50.399864: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 3 nodes (-482), 2 edges (-498), time = 45177.332ms.
2020-06-14 22:48:50.399884: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 3 nodes (0), 2 edges (0), time = 1.673ms.
2020-06-14 22:48:50.399897: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:786] Optimization results for grappler item: module_apply_default/TRTEngineOp_0_native_segment
2020-06-14 22:48:50.399909: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 485 nodes (0), 500 edges (0), time = 160.988ms.
2020-06-14 22:48:50.399929: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   layout: Graph size after: 485 nodes (0), 500 edges (0), time = 190.659ms.
2020-06-14 22:48:50.399942: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 485 nodes (0), 500 edges (0), time = 169.946ms.
2020-06-14 22:48:50.399960: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   TensorRTOptimizer: Graph size after: 485 nodes (0), 500 edges (0), time = 25.542ms.
2020-06-14 22:48:50.399980: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:788]   constant_folding: Graph size after: 485 nodes (0), 500 edges (0), time = 167.425ms.
```