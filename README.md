# HITBert
A fine-tuned BERT-based NLP model for classyfing HIT-related adverse events reports.

The model is based on 'emilyalsentzer/Bio_ClinicalBERT' from HuggingFace.
The model has been fine-tuned with the following hyperparameters on data extracted from the US MAUDE database.

batch_size = 16

activation_function = 'gelu'

freeze_layer_count = 8

attention_dropout_value = 0.5

hidden_dropout_value = 0.1

epochs = 7

learning_rate = 2e-5

# Disclaimer
When using this model, the provided Python code, or the dataset for any other projects, please cite the original work:

Luschi, A., Nesi, P., Iadanza, E. "Evidence-based Clinical Engineering: Health Information Technology Adverse Events Identification and Classification with Natural Language Processing", 2023
