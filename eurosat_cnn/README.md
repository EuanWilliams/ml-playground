## EuroSatCNN
- EuroSatCNN is a simple Convolutional Neural Network based image classifier implemented in PyTorch.
- It is trained on the EuroSAT dataset of different terrain types.

![Residential_9.jpg](eurosat_cnn/dataset/eurosat/2750/Residential/Residential_9.jpg)![SeaLake_8.jpg](eurosat_cnn/dataset/eurosat/2750/SeaLake/SeaLake_8.jpg)![River_3.jpg](eurosat_cnn/dataset/eurosat/2750/River/River_3.jpg)![PermanentCrop_7.jpg](eurosat_cnn/dataset/eurosat/2750/PermanentCrop/PermanentCrop_7.jpg)![Highway_14.jpg](eurosat_cnn/dataset/eurosat/2750/Highway/Highway_14.jpg)![AnnualCrop_43.jpg](eurosat_cnn/dataset/eurosat/2750/AnnualCrop/AnnualCrop_43.jpg)![Forest_9.jpg](eurosat_cnn/dataset/eurosat/2750/Forest/Forest_9.jpg)![Industrial_53.jpg](eurosat_cnn/dataset/eurosat/2750/Industrial/Industrial_53.jpg)![Pasture_11.jpg](eurosat_cnn/dataset/eurosat/2750/Pasture/Pasture_11.jpg)

# Results

### Iteration 0:
- Initial implementation.
- Total of 2 convolutional layers.
- Results:
  - Precision: 78%
  - Recall: 78%
  - Accuracy: 65%


![f1_per_class.png](models/eurosat_cnn_it_0/evaluation/f1_per_class.png)

### Iteration 1:
- Added a third convolutional layer.
- Big improvement.
- Results:
  - Precision: 92%
  - Recall: 92%
  - Accuracy: 85%

![f1_per_class.png](models/eurosat_cnn_it_1/evaluation/f1_per_class.png)


### Iteration 4:
- Update all ReLU activation functions to LeakyReLU.
- Minor improvement.
- Results:
  - Precision: 92.62%
  - Recall: 92.62%
  - Accuracy: 86.25%

![f1_per_class.png](models/eurosat_cnn_it_4/evaluation/f1_per_class.png)


### Iteration 6:
- Added a fourth convolutional layer.
- Minor improvement.
- Results:
  - Precision: 93.04%
  - Recall: 93.04%
  - Accuracy: 86.98%

![f1_per_class.png](models/eurosat_cnn_it_6/evaluation/f1_per_class.png)