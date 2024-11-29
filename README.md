# Food_Vision_101 

This is an **end-to-end CNN Image Classification Model** built on the Tranfer Learning technique which identifies the food in your image.

A pretrained Image Classification Model EfficientnetB1 was retrained on the infamous **Food101 Dataset**.

The Model actually beats the [**DeepFood**](https://arxiv.org/abs/1606.05675) Paper's model which had an Accuracy of  was **77.4%**  also trained on the same dataset but the interesting thing is, DeepFood's model took 2-3 days to train while our's was around 60min.

## **Key Highlights**

### **Dataset**
- **Source**: `Food101` dataset from TensorFlow Datasets.
- **Training Data**: 75,750 images.
- **Testing Data**: 25,250 images.

### **Model Architecture**
- **Base Model**: `EfficientNetB0` (pretrained on ImageNet).  
- **Approach**: Feature extraction followed by fine-tuning.  
- **Performance**: Outperforms DeepFood's benchmark with a **Top-1 Accuracy of 80.17%**.

### **Techniques Used**
1. **Mixed Precision Training**  
   - Achieves faster and memory-efficient computation by combining `float16` and `float32` data types.
2. **Transfer Learning**  
   - Utilizes pretrained EfficientNetB0 for feature extraction and fine-tuning.
3. **Optimized Data Pipeline**  
   - Employs `tf.data API` with **map**, **shuffle**, **batch**, and **prefetch** for efficient data loading.
4. **Callbacks for Enhanced Training**  
   - **Early Stopping**: Stops training if validation performance does not improve.  
   - **Learning Rate Reduction**: Dynamically adjusts the learning rate when progress stalls.  
   - **Model Checkpointing**: Saves the best-performing model during training.


> **Model :** `EfficientNetB1`
We're going to be building Food Vision Big™, using all of the data from the Food101 dataset.

Yep. All 75,750 training images and 25,250 testing images.

Two methods to significantly improve the speed of our model training:

Prefetching
Mixed precision training



## **Preprocessing the Data**

Since we've downloaded the data from TensorFlow Datasets, there are a couple of preprocessing steps we have to take before it's ready to model. 

More specifically, our data is currently:

* In `uint8` data type
* Comprised of all differnet sized tensors (different sized images)
* Not scaled (the pixel values are between 0 & 255)

Whereas, models like data to be:

* In `float32` data type
* Have all of the same size tensors (batches require all tensors have the same shape, e.g. `(224, 224, 3)`)
* Scaled (values between 0 & 1), also called normalized

To take care of these, we'll create a `preprocess_img()` function which:

* Resizes an input image tensor to a specified size using [`tf.image.resize()`](https://www.tensorflow.org/api_docs/python/tf/image/resize)
* Converts an input image tensor's current datatype to `tf.float32` using [`tf.cast()`](https://www.tensorflow.org/api_docs/python/tf/cast)

## **Building the Model : EfficientNetB1**

Implemented Mixed Precision training and Prefetching to decrease the time taken for the model to train.

### **Getting the Callbacks ready**
As we are dealing with a complex Neural Network (EfficientNetB0) its a good practice to have few call backs set up. Few callbacks I will be using throughtout this Notebook are :
 * **TensorBoard Callback :** TensorBoard provides the visualization and tooling needed for machine learning experimentation

 * **EarlyStoppingCallback :** Used to stop training when a monitored metric has stopped improving.
 
 * **ReduceLROnPlateau :** Reduce learning rate when a metric has stopped improving.

## **Performance Details**

| **Metric**            | **Feature Extraction** | **Fine-Tuning** |
|------------------------|------------------------|-----------------|
| **Train Accuracy**     | 72.41%                | 94.86%          |
| **Validation Accuracy**| 72.40%                | 80.17%          |
| **Test Accuracy**      | 72.79%                | 80.17%          |
| **Test Loss**          | 0.9993                | 0.9072          |

---

## **Results**

Our fine-tuned model achieved **80.17% top-1 accuracy**, surpassing the **DeepFood benchmark of 77.4%**. 

The exceptional performance was made possible by:
- **Mixed Precision Training**: Faster computations and efficient memory usage.  
- **Transfer Learning**: Leveraging pretrained EfficientNetB0 for better generalization.  
- **Data Pipeline Optimization**: Improved throughput using `tf.data API`.  

This approach significantly reduced training time while delivering superior accuracy, setting a new standard in food image classification. 🍔👁


### Evaluating the results
 
### Loss vs Epochs
 
![image](https://user-images.githubusercontent.com/61462986/202082223-83c3a8f2-26c9-455e-97d5-ee833a4b10cc.png)

### Accuracy vs Epochs

![image](https://user-images.githubusercontent.com/61462986/202082253-0d28ea8e-72af-4182-bf79-33b4119f27ef.png)

### Model's Class-wise Accuracy Score

 ![image](https://user-images.githubusercontent.com/61462986/202082047-6690d7cd-1999-4edc-9dc1-53fb9780ee89.png)

### Confusion Matrix

![image](https://user-images.githubusercontent.com/61462986/202082179-3337c5d7-fa06-4589-9050-1c2af1785808.png)

### Custom Prediction

![image](https://user-images.githubusercontent.com/61462986/202090045-4469ad2b-5366-41ed-8abc-1e5013b55ae6.png)



