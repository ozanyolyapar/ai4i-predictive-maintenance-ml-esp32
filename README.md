# ai4i-predictive-maintenance-ml-esp32

1. Run notebook (for exporting tflite version, running only MLP tensorflow cells might be sufficient. Or just download mlp_tf.tflite from releases)
2. `xxd -i mlp_tf.tflite > ../mlp_tf.h` (or just download mlp_tf.h from releases)
3. (optional) change wifi settings of esp32server
4. Install library `Chirale_TensorFlowLite` in Arduino IDE (currently the latest working version) 
5. Compile esp32server (this might take a while)

Alternatively, you can use pre-compiled binaries (under releases) to upload on esp32. 

# Screenshot from ESP32 server
![esp32 screenshot](https://raw.githubusercontent.com/ozanyolyapar/ai4i-predictive-maintenance-ml-esp32/refs/heads/main/images/esp32_screenshot.png)

# F1 scores (excl. MLP on ESP32)
![F1 scores](https://github.com/ozanyolyapar/ai4i-predictive-maintenance-ml-esp32/blob/main/images/f1.png?raw=true)
