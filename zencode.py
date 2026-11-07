import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from IPython.display import display
import PIL

data_dir = pathlib.Path('/content/drive/MyDrive/niku_exre')
print(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

img_height = 112
img_weidth = 112
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    image_size=(img_height, img_weidth),
    batch_size=batch_size,
    seed=200
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    image_size=(img_height, img_weidth),
    batch_size=batch_size,
    seed=200
)

class_names = train_ds.class_names
num_classes = len(class_names)

mdl = models.Sequential([
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),

    layers.Rescaling(1./255, input_shape=(img_height, img_weidth, 3)),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

mdl.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
mdl.summary()

epochs = 100
history = mdl.fit(train_ds, validation_data=val_ds, epochs=epochs)

#เช็กรูป1500รูปว่ามีรูปไหนทำนายผิดบ้าง
data_dir = pathlib.Path('/content/drive/MyDrive/niku_exre')
all_meat_images = list(data_dir.glob('*/*.jpg'))

# วนลูปเช็กทุกรูป
for idx, sample_img_path in enumerate(all_meat_images):
    img = tf.keras.utils.load_img(str(sample_img_path), target_size=(112, 112))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # ปิด Progress Bar
    predictions = mdl.predict(img_array, verbose=0)
    pred_label = class_names[np.argmax(predictions)]
    true_label = sample_img_path.parent.name

    # ถ้าทำนายผิด ให้แสดงผลแล้วทำงานต่อ
    if true_label != pred_label:
        print(f"รูปที่ (Index): {idx}")
        print("Predictions :", predictions)
        print("เฉลย:", true_label)
        print("ทำนายว่า:", pred_label)
        display(img)
        print("-" * 40) # เส้นแบ่งระหว่างรูป

data_dir = pathlib.Path('/content/drive/MyDrive/niku_exre')
all_meat_images = list(data_dir.glob('*/*.jpg'))
sample_img_path = all_meat_images[21]
img = tf.keras.utils.load_img(str(sample_img_path), target_size=(112, 112))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = mdl.predict(img_array)
print("Predictions :", predictions)
print("เฉลย:", sample_img_path.parent.name)
print("ทำนายว่า:", class_names[np.argmax(predictions)])

display(img)

mdl.save('zencode.keras')
