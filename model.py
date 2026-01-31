# Libraries for data manipulation and NLP
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

# NLP / Transformers
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# -------------------------------
# 1. Load datasets
# -------------------------------
def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = [line.strip().split(";") for line in lines]
    return pd.DataFrame(data, columns=["text", "emotion"])

train_df = load_data('train.txt')
val_df = load_data('val.txt')
test_df = load_data('test.txt')

# Optional: combine for plotting
full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# -------------------------------
# 2. Plot emotion distribution
# -------------------------------
palette = sns.color_palette("husl", full_df['emotion'].nunique())

plt.figure(figsize=(8,6))
ax = sns.countplot(data=full_df, x='emotion', palette=palette)
plt.xlabel("Emotion")
plt.ylabel("Count")

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='center', xytext=(0,5), textcoords='offset points')

plt.tight_layout()
plt.savefig("Figure_2_Emotion_Distribution.tiff", format="tiff", dpi=300, bbox_inches="tight")
plt.show()

# -------------------------------
# 3. Text cleaning
# -------------------------------
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    return text

for df in [train_df, val_df, test_df]:
    df['text'] = df['text'].apply(clean_text)

# -------------------------------
# 4. Tokenization
# -------------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_data(df):
    return tokenizer(df['text'].tolist(), padding=True, truncation=True, return_tensors="pt")

train_encodings = tokenize_data(train_df)
val_encodings   = tokenize_data(val_df)
test_encodings  = tokenize_data(test_df)

# -------------------------------
# 5. Map emotions to integers
# -------------------------------
emotion_to_id = {e: i for i, e in enumerate(train_df['emotion'].unique())}

train_labels = train_df['emotion'].map(emotion_to_id).values
val_labels   = val_df['emotion'].map(emotion_to_id).values
test_labels  = test_df['emotion'].map(emotion_to_id).values

# -------------------------------
# 6. Load BERT model
# -------------------------------
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(emotion_to_id))

# -------------------------------
# 7. Training arguments
# -------------------------------
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    seed=42,  # For reproducibility
)

# -------------------------------
# 8. Prepare datasets for Trainer
# -------------------------------
train_dataset = Dataset.from_dict({**train_encodings, 'labels': list(train_labels)})
val_dataset   = Dataset.from_dict({**val_encodings, 'labels': list(val_labels)})

# -------------------------------
# 9. Trainer initialization
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# -------------------------------
# 10. Train the model
# -------------------------------
trainer.train()

# -------------------------------
# 11. Evaluate on test set
# -------------------------------
test_dataset = Dataset.from_dict({**test_encodings, 'labels': list(test_labels)})
results = trainer.evaluate(test_dataset)
print("Test Results:", results)

# -------------------------------
# 12. Confusion matrix
# -------------------------------
# Generate predictions
preds_output = trainer.predict(test_dataset)
preds = np.argmax(preds_output.predictions, axis=1)
labels = test_labels

# Display confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(labels, preds, display_labels=list(emotion_to_id.keys()), cmap=plt.cm.Blues)
plt.title("Confusion Matrix of Emotion Classification")
plt.show()

# Optional: detailed classification report
print(classification_report(labels, preds, target_names=list(emotion_to_id.keys())))
