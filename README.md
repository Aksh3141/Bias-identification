# Bias-identification
We fine-tuned a ResNet-18 model on the FairFace dataset to predict gender labels, classifying images as either male or female. After training for 10 epochs, the model achieved 93.10% accuracy on the test set.

However, our analysis revealed that the model exhibits bias with respect to race. This means that the classification performance varies across different racial groups, indicating disparities in the model’s predictions. Further investigation into fairness metrics, such as demographic parity and equal opportunity, can help quantify and address these biases.


