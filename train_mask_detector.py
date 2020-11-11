# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator #
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input #
from tensorflow.keras.preprocessing.image import img_to_array #
from tensorflow.keras.preprocessing.image import load_img #
from tensorflow.keras.utils import to_categorical #
from sklearn.preprocessing import LabelBinarizer #
from sklearn.model_selection import train_test_split #
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt #
import numpy as np #
import os #

'''
DOCS section:

ImageDataGenerator : https://keras.io/api/preprocessing/image/#imagedatagenerator-class
mobilenetv2-function : https://keras.io/api/applications/mobilenet/#mobilenetv2-function
averagepooling2d-class : https://keras.io/api/layers/pooling_layers/average_pooling2d/#averagepooling2d-class
dropout-class : https://keras.io/api/layers/regularization_layers/dropout/#dropout-class
flatten-class : https://keras.io/api/layers/reshaping_layers/flatten/#flatten-class
dense : https://keras.io/api/layers/core_layers/dense/
input : https://keras.io/api/layers/core_layers/input/


'''
# initialize the initial learning rate, number of epochs to train for, and the batch size
INIT_LR = 1e-4
EPOCHS = 5
BS = 32

DIRECTORY = "./dataset"

CATEGORIES = ["with_mask", "without_mask"]

# take list of images from dataset , then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)

    	# important to create batches - adds extra para
    	image = preprocess_input(image)

    	# appending
    	data.append(image)
    	labels.append(category)

# labels
lb = LabelBinarizer()

# convert array into 2D array where row length depends on the total values it can take
# for example [0,1] will turn into [[1,0], [0,1]] but [0,1,2] will turn into [[1,0,0], [0,1,0], [0,0,1]]
labels = lb.fit_transform(labels)


labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

'''
This stratify parameter makes a split so that the proportion of values in the sample produced
 will be the same as the proportion of values provided to parameter stratify.
For example, if variable y is a binary categorical variable with values 0 and 1 and 
there are 25% of zeros and 75% of ones, 
stratify=y will make sure that your random split has 25% of 0's and 75% of 1's.
To remove unbalanced Dataset
'''
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")