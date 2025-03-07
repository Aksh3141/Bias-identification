# Bias Identification with FairFace Dataset

A ResNet-18 model was fine-tuned on the FairFace dataset to predict gender labels (Male/Female). After training for 10 epochs, the model achieved **93.10% accuracy** on the test set. However, our analysis revealed biases with respect to race.

## Bias Analysis Results

| Race | Accuracy (%) | Demographic Parity (P(Female)) | Equal Opportunity (TPR for Female) |
|------|------------|-------------------------------|-----------------------------------|
| 0    | 93.16      | 0.4716                        | 0.9043                            |
| 1    | 92.76      | 0.4336                        | 0.8910                            |
| 2    | 94.64      | 0.4886                        | 0.9253                            |
| 3    | 93.07      | 0.4622                        | 0.9088                            |
| 4    | 88.11      | 0.4383                        | 0.8283                            |
| 5    | 94.92      | 0.4789                        | 0.9253                            |
| 6    | 95.70      | 0.3160                        | 0.9167                            |

- **Accuracy**: The model performs differently across races, with the lowest accuracy observed for Race 4 (88.11%) and the highest for Race 6 (95.70%).
- **Demographic Parity**: Probability of classifying an image as "Female" varies by race, indicating bias in classification rates.
- **Equal Opportunity**: True positive rate (TPR) for Female is lower for some races (e.g., Race 4 at 0.8283), showing disparity in correct classification.

## Conclusion
The model demonstrates racial bias in gender classification, as indicated by variations in accuracy, demographic parity, and equal opportunity across different racial groups.



