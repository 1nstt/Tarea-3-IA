import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import seaborn as sns

# ============================
# CARGA DEL DATASET
# ============================
train_dir = "dataset_split/train"
val_dir = "dataset_split/val"
test_dir = "dataset_split/test"

IMG_SIZE = (64, 64)
BATCH = 32

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical'
)

val_gen = datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical'
)

test_gen = datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical', shuffle=False
)

class_names = list(train_gen.class_indices.keys())

# ============================
# TUNING DEL LEARNING RATE
# ============================
learning_rates = [1e-2, 1e-3, 1e-4]
best_lr = None
best_val_acc = -1

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(64, 64, 3)),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(5, activation="softmax")
    ])
    return model

print("\n=== TUNING DEL LEARNING RATE ===")
for lr in learning_rates:
    temp_model = build_model()
    temp_model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                       loss="categorical_crossentropy",
                       metrics=["accuracy"])

    print(f"\nProbando learning rate = {lr} ...")
    history = temp_model.fit(train_gen, validation_data=val_gen, epochs=3, verbose=0)

    val_acc = history.history["val_accuracy"][-1]
    print(f"Val_accuracy = {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_lr = lr

print(f"\n>>> Mejor learning rate encontrado: {best_lr}\n")

# ============================
# ENTRENAMIENTO FINAL
# ============================
model = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(best_lr),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(train_gen, validation_data=val_gen, epochs=20)

# ============================
# EVALUACIÓN EN TEST
# ============================
test_loss, test_acc = model.evaluate(test_gen)
print("\n===============================")
print(f"Accuracy en conjunto de prueba: {test_acc:.4f}")

# ----- F1-MACRO -----
y_true = test_gen.classes
y_pred = np.argmax(model.predict(test_gen), axis=1)
f1 = f1_score(y_true, y_pred, average="macro")
print(f"F1-macro: {f1:.4f}")

# ----- MATRIZ DE CONFUSIÓN -----
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Matriz de Confusión - Modelo Dropout")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig("confusion_dropout.png")
plt.close()

print("===============================\n")

# ============================
# GRÁFICAS ACCURACY / LOSS
# ============================

plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy - Modelo con Dropout")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig("accuracy_dropout.png")
plt.close()

plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss - Modelo con Dropout")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("loss_dropout.png")
plt.close()
