# handwritten-mathematical-symbols
1. Identifying handwritten mathematical symbols using deep neural networks and convolution. 
2. Saving and converting trained model into tflite (ready to run on mobile as well as edge devices).

Link of dataset:
  https://www.kaggle.com/guru001/hasyv2
(143 MB)

Context
HASYv2 - Handwritten Symbol database

Content
HASY contains 32px x 32px images of 369 symbol classes. In total, HASY contains over 150,000 instances of handwritten symbols.

This is basically a multiclass classification problem which I solved using CNN followed by Max pooling and dense layers.

*Folder and files arrangement*

Hand_written
  - handwritten_mathematical_model.ipynb
  - hasy-data
  - hasy-data-labels
  - symbols.csv
  - hasyv2
    - hasy-data
    - hasy-data-labels
    - symbols.csv
    - verification-task
   - verification-task

**Sample Images**

<p>
<img src = "Sample_images/v2-02130.png" width="40" height="40">
<img src = "Sample_images/v2-02362.png" width="40" height="40">
<img src = "Sample_images/v2-03594.png" width="40" height="40">
<p />

**Sample Labels**

<img src = "Label_samples/sample-2.png" width="360" height="160">
<img src = "Label_samples/sample-4.png" width="360" height="160">

