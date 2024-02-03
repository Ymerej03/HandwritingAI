import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage

from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CWERMetric

from model import train_model
from configs import ModelConfigs

import os

# based on https://github.com/pythonlessons/mltu/tree/main/Tutorials/03_handwriting_recognition and
# https://www.youtube.com/watch?v=WhRC31SlXzA same code ones more of a guide
# the code here is almost certainly worse and probably over commented
dataset_path = f"jeremy_words_dataset{os.sep}jeremy_words"
dataset_words = f"jeremy_words_dataset{os.sep}jeremy_words.txt"
dataset = []
vocab = set()
max_len = 0

# read in the files according to the IAM organisation adding the images and labels to the model
with open(dataset_words, "r") as text:
    line = "not blank"
    while line != "":
        # reads the line first so that when it continues it doesnt get stuck in an infinite loop
        line = text.readline()
        if line == "":
            continue
        # if it starts with a # it is a comment and is to be ignored
        if line.startswith("#"):
            continue
        split_line = line.split(" ")

        if split_line[1] == "err":
            continue
        # first item split on - is top level folder
        top_folder = split_line[0].split("-")[0]

        # first two items split on - but reconnected with - is next folder
        sub_folder = "-".join([split_line[0].split("-")[0], split_line[0].split("-")[1]])

        # first item split on whitespace + .png is image name
        image_name = split_line[0] + ".png"

        # last item split on whitespace is image transcription removing the newline that comes with it also
        transcription = split_line[-1].rstrip('\n')

        # getting the relative path to the image
        relative_path = os.path.join(dataset_path, top_folder, sub_folder, image_name)

        dataset.append([relative_path, transcription])
        vocab.update(list(transcription))
        max_len = max(max_len, len(transcription))

# Below is no longer my code.

# Create a ModelConfigs object to store model configurations
configs = ModelConfigs()

# Save vocab and maximum text length to configs
configs.vocab = "".join(vocab)
configs.max_text_length = max_len
configs.save()

# Create a data provider for the dataset
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
        ],
)

# Split the dataset into training and validation sets
train_data_provider, val_data_provider = data_provider.split(split = 0.9)

# Augment training data with random brightness, rotation and erode/dilate
train_data_provider.augmentors = [
    RandomBrightness(),
    RandomErodeDilate(),
    RandomSharpen(),
    RandomRotate(angle=10),
    ]

# Creating TensorFlow model architecture
model = train_model(
    input_dim = (configs.height, configs.width, 3),
    output_dim = len(configs.vocab),
)

# Compile the model and print summary
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
    loss=CTCloss(),
    metrics=[CWERMetric(padding_token=len(configs.vocab))],
)
model.summary(line_length=110)

# Define callbacks
earlystopper = EarlyStopping(monitor="val_CER", patience=20, verbose=1)
checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5", monitor="val_CER", verbose=1, save_best_only=True, mode="min")
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10, patience=10, verbose=1, mode="auto")
model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

# Train the model
model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx],
    workers=configs.train_workers
)

# Save training and validation datasets as csv files
train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))
