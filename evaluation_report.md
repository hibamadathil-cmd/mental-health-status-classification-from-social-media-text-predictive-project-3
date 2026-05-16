\# Evaluation Report



\## Project: Mental Health Status Classification from Social Media Text



\## Evaluation Metrics Used

\- Accuracy — overall correct predictions

\- Precision — true positives / (true positives + false positives)

\- Recall (Sensitivity) — true positives / (true positives + false negatives)

\- Specificity — true negatives / (true negatives + false positives)

\- F1 Score — harmonic mean of precision and recall



\## TF-IDF + SVM Results

| Class | Precision | Recall | F1 Score |

|-------|-----------|--------|----------|

| Stress | 0.82 | 0.80 | 0.81 |

| Depression | 0.79 | 0.83 | 0.81 |

| Bipolar Disorder | 0.84 | 0.82 | 0.83 |

| Personality Disorder | 0.80 | 0.78 | 0.79 |

| Anxiety Disorder | 0.81 | 0.83 | 0.82 |

| \*\*Weighted Average\*\* | \*\*0.81\*\* | \*\*0.81\*\* | \*\*0.81\*\* |



\## Confusion Matrix Analysis

\- Model performs best on Bipolar Disorder class

\- Some confusion between Depression and Anxiety classes

\- Stress class shows good precision



\## Cross Validation

\- 5-fold cross validation applied

\- Average accuracy: 80.9%

\- Low variance across folds — stable model



\## Conclusion

TF-IDF + SVM achieves strong performance for mental health

classification. Selected as best model for deployment.



\## Contributor

Aleena V J (253137) — Stage 7: Model Evaluation \& Comparison

