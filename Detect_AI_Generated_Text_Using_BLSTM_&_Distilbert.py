# %% [markdown]
# # Detect AI Generated Text Using BLSTM & Distilbert
# [Link](https://www.kaggle.com/code/shahbodsobhkhiz/detect-ai-generated-text-using-blstm-distilbert)

# %%
import tensorflow as tf
#plot tools
import seaborn as sns
import matplotlib.pyplot as plt
#simple tools
import numpy as np
import pandas as pd

# %%
DATA_PATH = 'D:/NTHU_NLP_2024_Term_Project_35/'
train_essays = pd.read_csv(f'{DATA_PATH}/train_essays.csv')
#train_65 = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text-dataset/Training_Essay_Data.csv')
train_65 = pd.read_csv(f'{DATA_PATH}/Training_Essay_Data.csv')
#prompts = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/train_prompts.csv')
prompts = pd.read_csv(f'{DATA_PATH}/train_prompts.csv')
#original = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_essays.csv")
original = pd.read_csv(f'{DATA_PATH}/train_essays.csv')
#train_v2 = pd.read_csv('/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv')
train_v2 = pd.read_csv(f'{DATA_PATH}/train_v2_drcat_02.csv')

#train_lim = pd.read_csv('/kaggle/input/llm-generated-essays/ai_generated_train_essays.csv')
train_lim = pd.read_csv(f'{DATA_PATH}/ai_generated_train_essays.csv')
#train_lim2 = pd.read_csv('/kaggle/input/llm-generated-essays/ai_generated_train_essays_gpt-4.csv')
train_lim2 = pd.read_csv(f'{DATA_PATH}/ai_generated_train_essays_gpt-4.csv')

# %%
original.head()

# %%
#Combining the datasets: 
combined_from_comp = pd.merge(original,prompts, on='prompt_id', how='left')

# %%
# Define the new column order
new_column_order = ['id', 'prompt_id' , 'prompt_name', 'instructions', 'source_text', 'text', 'generated']

# Reorder the columns
combined_from_comp = combined_from_comp[new_column_order]

# %%
train_lim = pd.concat([train_lim, train_lim2], ignore_index=True)

# %%
print("Training labels distribution:", np.bincount(train_lim['generated']))

# %%
print("Training labels distribution:", np.bincount(combined_from_comp['generated']))

# %%
combined_from_comp.head()

# %%
train_lim.head()

# %%
# Define the mapping dictionary
prompt_mapping = {
    0: 'car-free-cities',
    1: 'Does the electoral college work?'
}
instructions = {
    0 : 'Write an explanatory essay to inform fellow citizens about the advantages of limiting car usage. Your essay must be based on ideas and information that can be found in the passage set. Manage your time carefully so that you can read the passages; plan your response; write your response; and revise and edit your response. Be sure to use evidence from multiple sources; and avoid overly relying on one source. Your response should be in the form of a multiparagraph essay. Write your essay in the space provided.',
    1 : 'Write a letter to your state senator in which you argue in favor of keeping the Electoral College or changing to election by popular vote for the president of the United States. Use the information from the texts in your essay. Manage your time carefully so that you can read the passages; plan your response; write your response; and revise and edit your response. Be sure to include a claim; address counterclaims; use evidence from multiple sources; and avoid overly relying on one source. Your response should be in the form of a multiparagraph essay. Write your response in the space provided.'    
}
source_text = {
    0 : str(combined_from_comp['instructions'].unique()[0]),
    1 : str(combined_from_comp['instructions'].unique()[1])
}

# Apply the mapping to create the new column
train_lim['prompt_name'] = train_lim['prompt_id'].map(prompt_mapping)
train_lim['source_text'] = train_lim['prompt_id'].map(source_text)
train_lim['instructions'] = train_lim['prompt_id'].map(instructions)

# %%
train_lim = train_lim[new_column_order]

# %%
train_lim.head()

# %%
train_merged = pd.concat([combined_from_comp,train_lim],ignore_index = True)

# %%
train_merged.head()

# %%
train_merged.shape

# %%
print("Training labels distribution:", np.bincount(train_merged['generated']))

# %%
train_merged = train_merged[['text' , 'generated']]

# %%
train_v2 = train_v2[['text' , 'label']]
train_v2.rename(columns = {'label' : 'generated'} , inplace = True)

# %%
train_merged2 = pd.concat([train_v2,train_65] , ignore_index = True)

# %%
train_merged2.head()

# %%
train_merged = pd.concat([train_merged , train_merged2] , ignore_index = True)

# %%
print(np.bincount(train_merged['generated']))

# %%
# Shuffle the DataFrame
train_merged = train_merged.sample(frac=1, random_state=42).reset_index(drop=True)
train_merged = train_merged.head(15000)

# %%
print("Training labels distribution:", np.bincount(train_merged['generated']))

# %%
train_merged['word_count'] = train_merged['text'].apply(lambda x: len(str(x).split()))

# Calculate the statistics
plt.figure(figsize=(8, 6))
sns.violinplot(x=train_merged['word_count'])
plt.title('Violin Plot of Word Count per Tweet')
plt.xlabel('Word Count')
plt.show()

# %%
train_merged = train_merged.drop('word_count', axis = 1)

# %%
train_merged.head()

# %%
print("Training labels distribution:", np.bincount(train_merged['generated']))

# %% [markdown]
# ## Balacing the data labels here:

# %%
import pandas as pd
from sklearn.utils import resample

# Assuming 'train_merged' is your DataFrame and 'generated' is the label column
majority_class = train_merged[train_merged['generated'] == 0]
minority_class = train_merged[train_merged['generated'] == 1]

# %%
# Downsample the majority class
majority_downsampled = resample(majority_class,
                                replace=False,  # Sample without replacement
                                n_samples=len(minority_class),  # Match the number of minority class
                                random_state=42)  # For reproducibility

# %%
train_merged = pd.concat([majority_downsampled, minority_class])

# %%
print("Training labels distribution:", np.bincount(train_merged['generated']))

# %%
from sklearn.model_selection import train_test_split

# First, split into train+val and test
train_val_df, test_df = train_test_split(train_merged, test_size=0.1, random_state=42, stratify=train_merged['generated'])

# Now, split the train+val into train and validation
train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42, stratify=train_val_df['generated'])

# Print the sizes of the splits to verify
print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# %%
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# %%
train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_df['text'].tolist(), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=512)

# %%
# Convert encodings and labels into TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_df['generated'].tolist()))
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_df['generated'].tolist()))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_df['generated'].tolist()))

# Batch the datasets
train_dataset = train_dataset.shuffle(len(train_dataset)).batch(8)
val_dataset = val_dataset.batch(16)
test_dataset = test_dataset.batch(16)

# %%
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 2)

# Compile the model
class DistilBertWithDropout(tf.keras.Model):
    def __init__(self, base_model, num_labels, dropout_rate=0.3):
        super(DistilBertWithDropout, self).__init__()
        self.base_model = base_model  # Assign the base_model correctly
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.classifier = tf.keras.layers.Dense(num_labels, activation='softmax')

    def call(self, inputs, training=False):
        outputs = self.base_model(inputs)
        
        if len(outputs[0].shape) == 3:
            # Assume the output is [batch_size, sequence_length, hidden_size]
            pooled_output = outputs[0][:, 0, :]  # Extract the [CLS] token's output
        elif len(outputs[0].shape) == 2:
            # Assume the output is [batch_size, hidden_size] directly
            pooled_output = outputs[0]
        else:
            raise ValueError("Unexpected output shape from the base model.")
        
        dropout_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(dropout_output)
        return logits

# Instantiate the model
num_labels = 2  # Adjust this depending on your number of classes
dropout_rate = 0.3  # You can adjust the dropout rate
model = DistilBertWithDropout(model, num_labels, dropout_rate)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# %%
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("best_model", save_best_only=True, save_format='tf')
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

# %%
# Train the model with callbacks
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=4,  # You can set this to a higher number
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# %%
# Retrieve a list of accuracy results on training and validation data
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Retrieve a list of loss results on training and validation data
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Get the number of epochs
epochs = range(1, len(train_acc) + 1)

# Plot training and validation accuracy values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss values
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# %%
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# %%
# Make predictions on the test set
y_pred = model.predict(test_dataset)

# Since the model outputs logits, convert these to predicted class labels
y_pred_labels = tf.argmax(y_pred, axis=1)

# %%
# Extract true labels from the test set
y_true = []
for _, labels in test_dataset:
    y_true.extend(labels.numpy())

# Convert to a TensorFlow tensor for compatibility
y_true = tf.convert_to_tensor(y_true)

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Manually define the class labels if not available in the model config
class_labels = ['negative', 'positive']  # Replace with your actual class names

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred_labels)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# %%
# Save the custom model
model.save('./AI-detector', save_format='tf')

# Save the tokenizer
tokenizer.save_pretrained('./AI-detector-tokenizer')

# %%
loaded_model = tf.keras.models.load_model('./AI-detector')

loaded_tokenizer = DistilBertTokenizer.from_pretrained('./AI-detector-tokenizer')

# %%
train_df.shape

# %%
y_train = train_df['generated'].values

# %%
from gensim.models import Word2Vec

Embedding_dimensions = 150

# Creating Word2Vec training dataset.
Word2vec_train_data = list(map(lambda x: x.split(), train_df['text']))

# %%
# Defining the model and training it.
word2vec_model = Word2Vec(
    sentences=Word2vec_train_data,  # The tokenized training data
    vector_size=Embedding_dimensions,  # Size of the embedding vectors
    window=5,  # Maximum distance between the current and predicted word within a sentence
    min_count=5,  # Ignores all words with total frequency lower than this
    workers=8,  # Number of CPU cores to use for training
    sg=0  # Use Skip-Gram model (sg=1), or CBOW model (sg=0)
)

print("Vocabulary Length:", len(word2vec_model.wv.key_to_index))

# %%
input_length = 512

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# %%
vocab_length = 20000

tokenizer = Tokenizer(filters="", lower=False, oov_token="<oov>")
tokenizer.fit_on_texts(train_df['text'])
tokenizer.num_words = vocab_length
print("Tokenizer vocab length:", vocab_length)

# %%
X_train = pad_sequences(tokenizer.texts_to_sequences(train_df['text']), maxlen=input_length)
X_test  = pad_sequences(tokenizer.texts_to_sequences(test_df['text']) , maxlen=input_length)

print("X_train.shape:", X_train.shape)
print("X_test.shape :", X_test.shape)

# %%
filtered_word_index = {word: token for word, token in tokenizer.word_index.items() if token <= vocab_length}

# %%
embedding_matrix = np.zeros((vocab_length , Embedding_dimensions))

for word, token in filtered_word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[token] = word2vec_model.wv[word]

print("Embedding Matrix Shape:", embedding_matrix.shape)

# %%
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, Dense, LSTM, Conv1D, Embedding

# %%
def getModel():
    embedding_layer = Embedding(input_dim = vocab_length,
                                output_dim = Embedding_dimensions,
                                weights=[embedding_matrix],
                                input_length=input_length,
                                trainable=False)

    model = Sequential([
        embedding_layer,
        Bidirectional(LSTM(100, dropout=0.3, return_sequences=True)),
        Bidirectional(LSTM(100, dropout=0.3, return_sequences=True)),
        Conv1D(100, 5, activation='relu'),
        GlobalMaxPool1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid'),
    ],
    name="Sentiment_Model")
    return model

# %%
training_model = getModel()
training_model.summary()

# %%
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
             EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)]

# %%
training_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# %%
history = training_model.fit(
    X_train, y_train,
    batch_size=256,
    epochs=10,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1,
)

# %%
acc,  val_acc  = history.history['accuracy'], history.history['val_accuracy']
loss, val_loss = history.history['loss'], history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# %%
from sklearn.metrics import confusion_matrix, classification_report

def ConfusionMatrix(y_pred, y_test):
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    categories  = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
                xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)

# %%
X_test = test_df['text']
y_test = test_df['generated']

# Tokenize and pad the test data
X_test_sequences = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=input_length)

# Make predictions on the test data
y_pred = training_model.predict(X_test_sequences)

# Convert prediction probabilities to binary outcomes (0 or 1)
y_pred = np.where(y_pred >= 0.5, 1, 0)

# Print the confusion matrix
ConfusionMatrix(y_pred, y_test)

# Additionally, print the classification report for detailed metrics
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# %%
# Save the BiLSTM model
training_model.save('saved_bilstm_model.keras')

# Save the Word2Vec model
word2vec_model.save('word2vec_model.model')

# %%
from tensorflow.keras.models import load_model
loaded_bilstm_model = load_model('saved_bilstm_model.keras')
loaded_word2vec_model = Word2Vec.load('word2vec_model.model')

# %%
test_essay = pd.read_csv(f'{DATA_PATH}/test_essays.csv')
#test_essay = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')

# %%
test_essay.head()

# %%
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
test_encodings = tokenizer(test_essay['text'].tolist(), truncation=True, padding=True, max_length=512)

# %%
# Create the TensorFlow dataset only for these 3 entries
test_dataset = tf.data.Dataset.from_tensor_slices(dict(test_encodings)).batch(8)


# Make predictions using the model
predictions = model.predict(test_dataset)

# Convert logits to probabilities using softmax
prediction_probabilities = tf.nn.softmax(predictions, axis=-1).numpy()

# Extract the probability of the positive class
positive_class_probs = prediction_probabilities[:, 1]

# %%
print(f"Length of test_essay['id']: {len(test_essay['id'])}")
print(f"Length of positive_class_probs: {len(positive_class_probs)}")

# %%
results_df = pd.DataFrame({
    'id': test_essay['id'],
    'generated': positive_class_probs
})

# Save the results to a CSV file
results_df.to_csv('submission.csv', index=False)

# %%
print(results_df)


