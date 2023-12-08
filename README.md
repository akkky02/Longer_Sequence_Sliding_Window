# Various Experiments Notebook

## Description

This Jupyter Notebook provides a guide on how to handle longer sequences using a sliding window approach with the Hugging Face Transformers library. When working with Transformer models, there is often a maximum token limit that the model can accept, and longer sequences may need to be split into smaller chunks. The sliding window approach is a technique to process longer sequences by sliding a window of fixed size through the input.

## Conclusion

In conducting two experiments on a multiclass classification task for longer sequence, Experiment 1, characterized by a smaller effective batch size and fewer gradient accumulation steps, outperformed Experiment 2 in terms of accuracy, f1_macro, and f1_weighted metrics. The superior performance of Experiment 1 suggests that a more nuanced handling of the imbalanced dataset was achieved, reflected in higher sensitivity to minority classes. Both experiments benefited from advanced training techniques such as mixed precision training (fp16) and gradient checkpointing, highlighting the importance of modern practices in optimizing deep learning model training. While these results showcase the efficacy of Experiment 1, further exploration with diverse hyperparameter configurations, including two epochs, and a different model (bert-base-uncased) is done in the additional experiments to see if we can get an increase in the accuracy and f1_macro scores.

