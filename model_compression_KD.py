import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Conv3DTranspose, Dropout, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import segmentation_models_3D as sm
import matplotlib.pyplot as plt
import random

# ----------- Memory and Environment Setup -----------
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ----------- Data Loader with Normalization -----------
def load_img(img_dir, img_list):
    images = []
    for image_name in img_list:
        if image_name.endswith('.npy'):
            image = np.load(os.path.join(img_dir, image_name)).astype('float32')
            # MinMax normalization per modality
            for m in range(image.shape[-1]):
                vmin, vmax = np.percentile(image[..., m], 1), np.percentile(image[..., m], 99)
                image[..., m] = np.clip((image[..., m] - vmin) / (vmax - vmin + 1e-8), 0, 1)
            images.append(image)
    return np.array(images)

def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])
            yield (X, Y)
            batch_start += batch_size
            batch_end += batch_size

# ----------- Teacher Model Loading -----------
teacher = load_model(
    'brats_3d.hdf5',
    custom_objects={
        'dice_loss_plus_1focal_loss': sm.losses.DiceLoss() + sm.losses.CategoricalFocalLoss(),
        'iou_score': sm.metrics.IOUScore(threshold=0.5)
    }
)
teacher.trainable = False

# ----------- Improved Student Model (Deeper UNet, 3 channels) -----------
def improved_student_unet(IMG_HEIGHT=128, IMG_WIDTH=128, IMG_DEPTH=128, IMG_CHANNELS=3, num_classes=4):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS), dtype='float32')
    # Encoder
    c1 = Conv3D(32, (3,3,3), activation='relu', padding='same')(inputs)
    c1 = Conv3D(32, (3,3,3), activation='relu', padding='same')(c1)
    p1 = MaxPooling3D((2,2,2))(c1)
    c2 = Conv3D(64, (3,3,3), activation='relu', padding='same')(p1)
    c2 = Conv3D(64, (3,3,3), activation='relu', padding='same')(c2)
    p2 = MaxPooling3D((2,2,2))(c2)
    # Bottleneck
    c3 = Conv3D(128, (3,3,3), activation='relu', padding='same')(p2)
    c3 = Dropout(0.3)(c3)
    c3 = Conv3D(128, (3,3,3), activation='relu', padding='same')(c3)
    # Decoder
    u4 = Conv3DTranspose(64, (2,2,2), strides=(2,2,2), padding='same')(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv3D(64, (3,3,3), activation='relu', padding='same')(u4)
    c4 = Conv3D(64, (3,3,3), activation='relu', padding='same')(c4)
    u5 = Conv3DTranspose(32, (2,2,2), strides=(2,2,2), padding='same')(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv3D(32, (3,3,3), activation='relu', padding='same')(u5)
    c5 = Conv3D(32, (3,3,3), activation='relu', padding='same')(c5)
    outputs = Conv3D(num_classes, (1,1,1), activation='softmax')(c5)
    return Model(inputs, outputs)

student = improved_student_unet()

# ----------- Precompute Teacher Predictions -----------
def save_teacher_predictions(teacher_model, img_dir, img_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for image_name in img_list:
        if image_name.endswith('.npy'):
            img = np.load(os.path.join(img_dir, image_name)).astype('float32')
            pred = teacher_model.predict(img[np.newaxis, ...])[0].astype('float32')
            np.save(os.path.join(output_dir, image_name), pred)

# Set paths
train_img_dir = r"D:\mini_pro\mp_dataset\input_data_128\train\images"
train_mask_dir = r"D:\mini_pro\mp_dataset\input_data_128\train\masks"
val_img_dir = r"D:\mini_pro\mp_dataset\input_data_128\val\images"
val_mask_dir = r"D:\mini_pro\mp_dataset\input_data_128\val\masks"
train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)
val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

# Only run ONCE to precompute teacher predictions
# save_teacher_predictions(teacher, train_img_dir, train_img_list, "teacher_preds/train/")
# save_teacher_predictions(teacher, val_img_dir, val_img_list, "teacher_preds/val/")

# ----------- KD Data Generator (Tuple output, float32) -----------
class KDDatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_dir, img_list, mask_dir, mask_list, teacher_pred_dir, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.teacher_pred_dir = teacher_pred_dir
        self.img_list = img_list
        self.mask_list = mask_list
        self.batch_size = batch_size

    def __len__(self):
        return len(self.img_list) // self.batch_size

    def __getitem__(self, idx):
        batch_imgs = []
        batch_masks = []
        batch_teacher = []
        batch_start = idx * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self.img_list))
        for i in range(batch_start, batch_end):
            img = np.load(os.path.join(self.img_dir, self.img_list[i])).astype('float32')
            # Normalize again for safety
            for m in range(img.shape[-1]):
                vmin, vmax = np.percentile(img[..., m], 1), np.percentile(img[..., m], 99)
                img[..., m] = np.clip((img[..., m] - vmin) / (vmax - vmin + 1e-8), 0, 1)
            mask = np.load(os.path.join(self.mask_dir, self.mask_list[i])).astype('float32')
            teacher_pred = np.load(os.path.join(self.teacher_pred_dir, self.img_list[i])).astype('float32')
            batch_imgs.append(img)
            batch_masks.append(mask)
            batch_teacher.append(teacher_pred)
        return (
            np.array(batch_imgs),
            (np.array(batch_masks), np.array(batch_teacher))
        )

# ----------- Training Configuration -----------
batch_size = 1  # Increase if you have more memory
train_kd_gen = KDDatasetGenerator(train_img_dir, train_img_list, train_mask_dir, train_mask_list, "teacher_preds/train/", batch_size)
val_kd_gen = KDDatasetGenerator(val_img_dir, val_img_list, val_mask_dir, val_mask_list, "teacher_preds/val/", batch_size)
steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

# ----------- Student KD Model -----------
class StudentKDModel(Model):
    def __init__(self, student):
        super().__init__()
        self.student = student

    def compile(self, optimizer, metrics, alpha=0.3, temperature=2.0):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.alpha = alpha
        self.temperature = temperature
        self.dice_loss = sm.losses.DiceLoss()
        self.focal_loss = sm.losses.CategoricalFocalLoss()
        self.kld = tf.keras.losses.KLDivergence()

    def train_step(self, data):
        x, (y_mask, y_teacher) = data
        with tf.GradientTape() as tape:
            y_pred = self.student(x, training=True)
            seg_loss = self.dice_loss(y_mask, y_pred) + self.focal_loss(y_mask, y_pred)
            y_pred_soft = tf.nn.softmax(y_pred / self.temperature)
            y_teacher_soft = tf.nn.softmax(y_teacher / self.temperature)
            kd_loss = self.kld(y_teacher_soft, y_pred_soft)
            total_loss = self.alpha * seg_loss + (1 - self.alpha) * kd_loss
        grads = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.compiled_metrics.update_state(y_mask, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, (y_mask, y_teacher) = data
        y_pred = self.student(x, training=False)
        seg_loss = self.dice_loss(y_mask, y_pred) + self.focal_loss(y_mask, y_pred)
        y_pred_soft = tf.nn.softmax(y_pred / self.temperature)
        y_teacher_soft = tf.nn.softmax(y_teacher / self.temperature)
        kd_loss = self.kld(y_teacher_soft, y_pred_soft)
        total_loss = self.alpha * seg_loss + (1 - self.alpha) * kd_loss
        self.compiled_metrics.update_state(y_mask, y_pred)
        return {m.name: m.result() for m in self.metrics}

student_kd = StudentKDModel(student)
student_kd.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=[sm.metrics.IOUScore(threshold=0.5)],
    alpha=0.3,
    temperature=2.0
)

checkpoint = ModelCheckpoint('best_student_kd.h5', save_best_only=True, monitor='val_iou_score', mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5)
early_stop = EarlyStopping(monitor='val_iou_score', patience=20, mode='max', restore_best_weights=True)

history = student_kd.fit(
    train_kd_gen,
    validation_data=val_kd_gen,
    epochs=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps_per_epoch,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

student_kd.student.save('student_kd_final.h5')

# ----------- Visualization (Optional) -----------
x, (y_true, y_teacher) = val_kd_gen[random.randint(0, len(val_kd_gen)-1)]
y_pred = student_kd.student.predict(x)
n_slice = 64
plt.figure(figsize=(18,6))
plt.subplot(131); plt.imshow(x[0,:,:,n_slice,0], cmap='gray'); plt.title('Flair')
plt.subplot(132); plt.imshow(np.argmax(y_true[0,:,:,n_slice], -1)); plt.title('Ground Truth')
plt.subplot(133); plt.imshow(np.argmax(y_pred[0,:,:,n_slice], -1)); plt.title('Student Prediction')
plt.show()
