# :dart: _Design, train, test a CNN network and apply *Regularization* methods to tackling Overfitting problem on [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)_




## *step 1:* _train and test model without Regularization_


#### - *model summary*



<img width="750" height="719" alt="base_cnn_model_summary" src="https://github.com/user-attachments/assets/c672c81d-9484-425b-8609-c675b146ca72" />



###### _train/validation loss/accuracy curves_ 



<img width="1000" height="500" alt="train_val_accuracy_without_regularization" src="https://github.com/user-attachments/assets/7e8b44b2-b9cc-4427-b9b5-b999ab7a480f" />

<img width="1000" height="500" alt="train_val_loss_without_regularization" src="https://github.com/user-attachments/assets/43cc966d-c222-4de6-b2e9-5f9c77249490" />




####  * Confusion Matrix before regularization *
 
 
<img width="800" height="800" alt="Base_Class_Confusion_Matrix" src="https://github.com/user-attachments/assets/ccb32c0b-b02a-453f-bbf7-5947c89778d0" />



## *step 2:* _train and test model along with Regularization_

---

 > #### Dropout layers:
> - It is used to prevent over-fitting. In such a way that we randomly ignore a percentage of neurons during network training. The large number of parameters of the network and the strong dependence between neurons cause the power of each neuron to be limited and overfit on the most data.


> #### Batch normalization:
> - Instead of normalizing only the initial input, normalize the input of all layers.One of the advantages of using this method is that the weight of some neurons does not increase too much.


> #### Augmentation:More training data for network training_




#### _Regularized model summary_

<img width="750" height="923" alt="regularized_cnn_model_summary" src="https://github.com/user-attachments/assets/ae06ec5d-c45e-4492-a3ff-22ca4a19710b" />





###### _train/validation loss/accuracy curves_

<img width="1000" height="500" alt="train_val_accuracy_with_regularization" src="https://github.com/user-attachments/assets/a23a7ecc-2396-4173-8dae-4db4c1de0575" />

<img width="1000" height="500" alt="train_val_loss_with_regularization" src="https://github.com/user-attachments/assets/76104ac0-7f63-482c-93f7-b3e875fc9d82" />



#### _Confusion Matrix after regularization_


<img width="800" height="800" alt="Regularized_Class_Confusion_Matrix" src="https://github.com/user-attachments/assets/5c46f3f3-1d5f-4a5e-9d3d-8ea4aa8ebf4d" />






