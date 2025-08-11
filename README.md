
# ML-Based IMU Orientation Estimator

This project demonstrates **orientation classification** (Pitch, Roll, Yaw) using a simple feedforward neural network in TensorFlow/Keras.  
It uses gyroscope sensor data stored in CSV files and trains a model to classify the orientation into one of three classes.
The system uses a trained ML model to process sensor readings (accelerometer, gyroscope, and magnetometer) and estimate the current orientation.

---

## Features
- Real-time IMU data capture.
- Preprocessing pipeline for Pitch, Roll and Yaw detection using raw Gyroscope Data
- ML model for orientation estimation.
- Support for exporting models to **TensorFlow Lite** format for deployment on embedded devices.



## Tech Stack
- **Python 3**
- **TensorFlow / Keras**
- **NumPy / Pandas**
- **Matplotlib** (for visualization)
- **Jupyter Notebook** for development and testing.



## Repository Structure
```
.
├── IMU_Orientation_Estimator.ipynb # Jupyter notebook
├── orientation_model.tflite     # TensorFlow Lite model 
├── data/   
│    ├── Combined.csv                 # Example IMU dataset       
│    ├── Pitch.csv 
│    ├── Roll.csv 
│    ├── Yaw.csv               
├── README.md                      
└── LICENSE                        
```



## Getting Started
### Clone the Repository
```bash
git clone https://github.com/yourusername/ML-IMU-Orientation-Estimator.git
cd ML-IMU-Orientation-Estimator
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

###  Train the Model
You can open the Jupyter Notebook and execute cells to train the model:
```bash
jupyter notebook Orientation_Estimator.ipynb
```



## Example Usage
```python
import tensorflow as tf
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="orientation_model.tflite")
interpreter.allocate_tensors()

# Prepare input data
imu_data = np.array([[ax, ay, az, gx, gy, gz, mx, my, mz]], dtype=np.float32)

# Run inference
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']
interpreter.set_tensor(input_index, imu_data)
interpreter.invoke()
orientation = interpreter.get_tensor(output_index)

print("Predicted Orientation:", orientation)
```


## License
This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.


## Author
- **Harshal Patil**
- GitHub: [HarshalP05](https://github.com/HarshalP05)

