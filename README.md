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

# Description
The model can be fine-tuned and tested from scratch by using the provided Python code for Jupyter Notebook (HITBert_kfold.ipynb). 
A ready-to-be-used version of the model is also available. This version has been fine-tuned, validated, and tested on the provided dataset. The testing metrics of the provided model are:

Accuracy: 0.9946

Precision: 0.9893

Recall: 1.0000

F1-score: 0.9946

MCC: 98.93%

The validation has been performed with the 10-fold validation technique. 
ROC and Confusion Matrix can be found in the repository.

# Implementation
The following Python code can be used to load the saved weights for the provided model and use it to classify new AE reports:
```
pip install transformers
```
```
pip install tensorflow
```
```
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline
```
```
# Load the model and the tokenizer
bert_model = 'emilyalsentzer/Bio_ClinicalBERT'
attention_dropout_value = 0.5
hidden_dropout_value = 0.1
activation_function = 'gelu'

tokenizer = AutoTokenizer.from_pretrained(bert_model, do_lower_case = True)
model = AutoModelForSequenceClassification.from_pretrained(
    bert_model,
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
    attention_probs_dropout_prob = attention_dropout_value,
    hidden_dropout_prob  = hidden_dropout_value,
    hidden_act = activation_function
)
```
```
cp = torch.load("HitBert.pt")
model.load_state_dict(cp['model_state_dict'])
```
```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    model.cuda()
```
```
text = "REPORTER INDICATED THAT THE SCREEN ON HIS VNS THERAPY PROGRAMMING HANDHELD WAS FREEZING DURING USE. TROUBLESHOOTING DID NOT RESOLVE THE ISSUE. GOOD FAITH ATTEMPTS TO OBTAIN BOTH FURTHER INFO AND THE HANDHELD FOR PRODUCT ANALYSIS ARE CURRENTLY BEING MADE. DEVICE MALFUNCTION IS SUSPECTED, BUT DID NOT CAUSE OR CONTRIBUTE TO DEATH OR SERIOUS INJURY."
```
```
explainer = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0, top_k=None)
explainer(text, padding="max_length", truncation=True, max_length=512, add_special_tokens = True)

# Output (LABEL_1 is for HIT class, LABEL_2 is for non-HIT class):
# [[{'label': 'LABEL_1', 'score': 0.9999047517776489},
#  {'label': 'LABEL_0', 'score': 9.521505126031116e-05}]]
```

# Disclaimer
When using this model, the provided Python code, or the dataset for any other projects, please cite the original work:

Luschi, A., Nesi, P., Iadanza, E. "Evidence-based Clinical Engineering: Health Information Technology Adverse Events Identification and Classification with Natural Language Processing", Heliyon, Vol. 9(11), 2023 [DOI: 10.1016/j.heliyon.2023.e21723]
