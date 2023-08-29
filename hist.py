import numpy as np
import pandas as pd
from glob import glob
from skimage.io import imread
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.nasnet import NASNetMobile
from keras.applications.xception import Xception
from keras.utils.vis_utils import plot_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Average, Input, Concatenate, GlobalMaxPooling2D
from keras.models import Model
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
import seaborn as sns
from skimage.io import imread
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# output files
TRAINING_LOGS_FILE = "training_logs.csv"
MODEL_SUMMARY_FILE = "model_summary.txt"
MODEL_PLOT_FILE = "model.png"
TRAINING_PLOT_FILE = "validation.png"
MODEL_FILE = "model.h5"  # Define the model file name

# Define the ModelCheckpoint callback function
checkpoint = ModelCheckpoint(MODEL_FILE, 
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             save_weights_only=False, 
                             mode='auto', 
                             save_freq='epoch')

# Hyperparams
SAMPLE_COUNT = 85000
TRAINING_RATIO = 0.9
IMAGE_SIZE = 96
EPOCHS = 30
BATCH_SIZE = 192
VERBOSITY = 1
TESTING_BATCH_SIZE = 5000

input_dir = "D:/PFE/histopathologicCancer/"
training_dir = os.path.join(input_dir, "train")
file_path = os.path.join(training_dir, "2d829b51214d3de2cc6d0b6551c5cdf14defb808.tif")
print("Start mapping files to classes and moving them to a new folder... Please wait...")
data_frame = pd.read_csv(os.path.join(input_dir, 'train_labels.csv'))
data_frame['path'] = data_frame['id'].apply(lambda x: os.path.join(training_dir, x + '.tif').replace('\\', '/'))
print(len(data_frame['id']))
print("Sample path inside the data frame: ", data_frame["path"][0])
data_frame['label'] = data_frame['label'].astype(str) # Convert label column to string type
data_frame['id'] = data_frame.path.map(lambda x: x.split('/')[-1].split('.')[0])

training_path = os.path.join(training_dir, "training")
validation_path = os.path.join(training_dir, "validation")


for folder in [training_path, validation_path]:
    for subfolder in ['0', '1']:
        path = os.path.join(folder, subfolder)
        os.makedirs(path, exist_ok=True)


training, validation = train_test_split(data_frame, train_size=TRAINING_RATIO, stratify=data_frame['label'])
data_frame.set_index('id', inplace=True)

# Create an instance of ImageDataGenerator
data_generator = ImageDataGenerator(rescale=1./255,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    rotation_range=90,
                                    zoom_range=0.1,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1)

# Define the training and validation data generators
training_data_generator = data_generator.flow_from_dataframe(dataframe=training,
                                                             x_col='path',
                                                             y_col='label',
                                                             batch_size=BATCH_SIZE,
                                                             shuffle=True,
                                                             class_mode='binary',
                                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                             subset='training')

validation_data_generator = data_generator.flow_from_dataframe(dataframe=validation,
                                                               x_col='path',
                                                               y_col='label',
                                                               batch_size=BATCH_SIZE,
                                                               shuffle=True,
                                                               class_mode='binary',
                                                               target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                               subset='validation')

# Define the model architecture
base_model = NASNetMobile(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights=None, pooling='avg')
x = Dropout(0.5)(base_model.output)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=x)



# Compile the model
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print the model summary
with open(MODEL_SUMMARY_FILE, 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))



# Train the model
history = model.fit(training_data_generator,
                    steps_per_epoch=len(training_data_generator),
                    validation_data=validation_data_generator,
                    validation_steps=len(validation_data_generator),
                    epochs=EPOCHS,
                    verbose=VERBOSITY,
                    callbacks=[checkpoint])



# Evaluate the model on the validation data
model = load_model(MODEL_FILE)  # Load the best weights from the ModelCheckpoint callback
validation_data_generator = data_generator.flow_from_dataframe(dataframe=validation,
                                                               x_col='path',
                                                               y_col='label',
                                                               batch_size=TESTING_BATCH_SIZE,
                                                               shuffle=False,
                                                               class_mode='binary',
                                                               target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                               subset='validation')

validation_pred = model.predict_generator(validation_data_generator, steps=len(validation_data_generator),
                                         verbose=VERBOSITY)
validation_labels = validation_data_generator.classes
validation_labels = validation_labels[:len(validation_pred)]
validation_auc = roc_auc_score(validation_labels, validation_pred)

# Predict on test data
testing_data_generator = data_generator.flow_from_dataframe(dataframe=testing,
                                                            x_col='path',
                                                            y_col='label',
                                                            batch_size=TESTING_BATCH_SIZE,
                                                            shuffle=False,
                                                            class_mode='binary',
                                                            target_size=(IMAGE_SIZE, IMAGE_SIZE))

testing_pred = model.predict_generator(testing_data_generator, steps=len(testing_data_generator), verbose=VERBOSITY)
testing_labels = testing_data_generator.classes

# Calculate test metrics
testing_pred_classes = np.round(testing_pred).flatten()
testing_accuracy = accuracy_score(testing_labels, testing_pred_classes)
testing_precision = precision_score(testing_labels, testing_pred_classes)
testing_recall = recall_score(testing_labels, testing_pred_classes)
testing_f1_score = f1_score(testing_labels, testing_pred_classes)
testing_auc = roc_auc_score(testing_labels, testing_pred)



# Image size for prediction
PREDICTION_IMAGE_SIZE = 96

# Function to predict the label of an image
def predict(image_path):
    # Load and preprocess the image
    image = load_img(image_path, target_size=(PREDICTION_IMAGE_SIZE, PREDICTION_IMAGE_SIZE))
    image_array = img_to_array(image)
    image_array = image_array / 255.0  # Normalize the image

    # Reshape the image array to match the model's input shape
    image_array = np.expand_dims(image_array, axis=0)

    # Perform the prediction
    prediction = model.predict(image_array)
    predicted_label = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'

    return predicted_label

# Example usage
image_path = 'D:/PFE/histopathologicCancer/image.tif'
predicted_label = predict(image_path)
tumor_detected = predicted_label == 'Negative'
# Set up the PDF object
pdf = FPDF()

# Add a new page to the PDF
pdf.add_page()

# Set the font and size for the chapter body text
pdf.set_font("Arial", size=10)
if tumor_detected:
    pdf.cell(0, 10, "I want to gently inform you that the results of your recent medical tests indicate the presence of histopathologic cancer.", ln=True)
    pdf.cell(0, 10, "I understand that this news may be overwhelming and evoke various emotions.", ln=True)
    pdf.cell(0, 10, "Please know that you are not alone, and we are here to provide you with the care and support you need.", ln=True)
    pdf.ln(10)
    pdf.cell(0, 10, "Cancer Type: Histopathologic Cancer", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 10, "Treatment Options:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, "1. Surgery: Surgical removal of the tumor is a common treatment option for histopathologic cancer.", ln=True)
    pdf.cell(0, 10, "   It involves removing the tumor and surrounding tissues to eliminate the cancer.", ln=True)
    pdf.cell(0, 10, "2. Radiation therapy: Radiation therapy uses high-energy rays to kill cancer cells and shrink tumors.", ln=True)
    pdf.cell(0, 10, "   It can be administered externally or internally, depending on the specific case.", ln=True)
    pdf.cell(0, 10, "3. Chemotherapy: Chemotherapy drugs are used to kill cancer cells throughout the body.", ln=True)
    pdf.cell(0, 10, "   It can be administered orally or intravenously, depending on the treatment plan.", ln=True)
    pdf.cell(0, 10, "4. Hormone therapy: Hormone therapy blocks or lowers the amount of hormones in the body", ln=True)
    pdf.cell(0, 10, "   to stop or slow down the growth of hormone receptor-positive histopathologic cancer.", ln=True)
    pdf.cell(0, 10, "5. Targeted therapy: Targeted therapy targets specific genes or proteins", ln=True)
    pdf.cell(0, 10, "   that are involved in the growth and survival of histopathologic cancer cells.", ln=True)
    pdf.cell(0, 10, "6. Immunotherapy: Immunotherapy helps the immune system recognize and attack cancer cells.", ln=True)
    pdf.ln(10)
    pdf.ln(10)
    pdf.set_text_color(255, 0, 0)
    pdf.set_font("Arial", "I", 13)
    pdf.set_xy(0, pdf.get_y())
    pdf.multi_cell(pdf.w, 10, 'Don\'t forget that: "Strength doesn\'t come from what you can do. It comes from overcoming the things you once thought you couldn\'t."', align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_text_color(0, 0, 255)
    pdf.set_font("Arial", "U", 10)
    pdf.cell(0, 10, "More information about Histopathologic Cancer:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "1. American Cancer Society - Histopathologic Cancer", ln=True, link="https://www.cancer.org/cancer/histopathologic-cancer")
    pdf.cell(0, 10, "2. National Cancer Institute - Histopathologic Cancer", ln=True, link="https://www.cancer.gov/types/histopathologic")
    pdf.cell(0, 10, "3. Cancer Research UK - Histopathologic Cancer", ln=True, link="https://www.cancerresearchuk.org/about-cancer/histopathologic-cancer")
else:
    pdf.cell(0, 10, "I would like to inform you that the results of your recent medical tests indicate that you do not have histopathologic cancer.", ln=True)
    pdf.ln(5)
    pdf.cell(0, 10, "However, it's important to note that there may be other conditions or diseases that could be causing your symptoms.", ln=True)
    pdf.cell(0, 10, "I recommend further evaluation to determine the underlying cause.", ln=True)
    pdf.cell(0, 10, "It would be beneficial to consult with a healthcare professional for a comprehensive examination and appropriate diagnostic tests.", ln=True)
    pdf.ln(5)
    pdf.cell(0, 10, "In the meantime, here are a few suggestions to improve your overall well-being:", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 10, "1. Adopt a Healthy Lifestyle:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 10, "Adopting a healthy lifestyle can contribute to better overall health.", ln=True)
    pdf.cell(0, 10, "Eat a balanced diet rich in fruits, vegetables, and whole grains. Engage in regular physical activity", ln=True)
    pdf.cell(0, 10, "and exercise to strengthen your body. Get enough restful sleep and manage stress effectively.", ln=True)
    pdf.cell(0, 10, "These lifestyle choices can support overall wellness and reduce the risk of various health problems.", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 10, "2. Seek Further Medical Evaluation:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, "Although you do not have histopathologic cancer, it is important to seek further medical evaluation", ln=True)
    pdf.cell(0, 10, "to identify the underlying cause of your symptoms.", ln=True)
    pdf.cell(0, 10, "Consult with a healthcare professional who can conduct comprehensive", ln=True)
    pdf.cell(0, 10,"evaluations and appropriate diagnostic tests to determine the underlying condition.", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 10, "3. Follow Healthcare Professionals' Advice:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, "It is crucial to follow the advice and recommendations provided by healthcare", ln=True)
    pdf.cell(0, 10, "professionals for a comprehensive assessment of your health condition.", ln=True)
    pdf.ln(5)
    pdf.cell(0, 10, "Please keep in mind that although you do not have histopathologic cancer,", ln=True)
    pdf.cell(0, 10, "it is essential to seek proper medical evaluation", ln=True)
    pdf.cell(0, 10, "and follow the advice of healthcare professionals", ln=True)
    pdf.cell(0, 10, "for a comprehensive assessment of your health condition.", ln=True)
    pdf.ln(5)
    pdf.set_text_color(255, 0, 0)
    pdf.set_font("Arial", size=11, style="B")
    pdf.cell(0, 10, "\"Health is a treasure that surpasses all riches.\" ", ln=True)

# Reset the text color to black
pdf.set_text_color(0, 0, 0)

# Save the PDF
pdf.output("histopathologic_cancer_report.pdf")

