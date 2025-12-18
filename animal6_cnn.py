# Animal-6 Classification (Simplified CNN) — Fully Commented
# Works with a folder structure like:
# Animal_Dataset/
# butterfly/ cat/ chicken/ elephant/ horse/ spider/
# No explicit train/val/test folders needed; we split 80/20 here.
# ============================================================
 # -----------------------
 # 0) Imports & utilities
 # -----------------------
 import os, sys, itertools, math
 import numpy as np
 import matplotlib.pyplot as plt
 import tensorflow as tf
 from tensorflow import keras
 from tensorflow.keras import layers
 print("TensorFlow:", tf.__version__)
 print("Python:", sys.version)
 # -----------------------
 # 1) Mount Google Drive
 # -----------------------
 # This lets Colab read the dataset you saved under My Drive.
 from google.colab import drive
 # If you previously mounted with partial permissions, unmount first:
try:
drive.mount('/content/drive') # Approve with full access (view/manage files)
 drive.flush_and_unmount()
 except Exception:
 pass
# -------------------------------------
# 2) Locate the dataset folder on Drive
# -------------------------------------
# We try to auto-find a folder named "Animal_Dataset" under MyDrive.
def find_path(root, target="Animal_Dataset", max_depth=4):
  for dirpath, dirnames, _ in os.walk(root):
 depth = dirpath[len(root):].count(os.sep)
 if depth > max_depth:
 # don't descend deeper to keep things fast
 del dirnames[:]
 continue
 if os.path.basename(dirpath) == target:
 return dirpath
 return None
 DATA_DIR = "/content/drive/MyDrive/EE267/Animal_Dataset"
)
print(" Using dataset at:", DATA_DIR)
 assert DATA_DIR and os.path.exists(DATA_DIR), (
 "Couldn't locate 'Animal_Dataset'. "
 "If you see this, set DATA_DIR manually to your folder path."
  # Peek at the top level so you can confirm the 6 class folders are there
 for root, dirs, files in itertools.islice(os.walk(DATA_DIR), 1):
 print(" ", root, "->", dirs[:6], f"(files: {len(files)})")
 # ----------------------------------------------
 # 3) Create training/validation (and test) sets
 # ----------------------------------------------
 # Since Animal_Dataset has 6 subfolders (one per class) and no explicit split,
 # we create an 80/20 train/val split on the fly.
 IMG_SIZE = (128, 128) # keep it modest for speed
 BATCH = 32
 SEED = 1337
 train_ds = keras.utils.image_dataset_from_directory(
 )
DATA_DIR,
 validation_split=0.2, subset="training", seed=SEED,
 image_size=IMG_SIZE, batch_size=BATCH
 val_ds = keras.utils.image_dataset_from_directory(
 DATA_DIR,
 validation_split=0.2, subset="validation", seed=SEED,
📂
✅

  image_size=IMG_SIZE, batch_size=BATCH
)
 # IMPORTANT: Save class_names BEFORE caching/prefetching
 # (after prefetch, the dataset becomes a PrefetchDataset without this attribute)
 class_names = train_ds.class_names
 num_classes = len(class_names)
 print(" Classes:", class_names)
 # For Problem 2 we can use validation as a proxy "test" set
 test_ds_raw = val_ds
 # Performance tweaks (cache + prefetch)
 AUTOTUNE = tf.data.AUTOTUNE
 train_ds = train_ds.cache().prefetch(AUTOTUNE)
 val_ds = val_ds.cache().prefetch(AUTOTUNE)
 test_ds = test_ds_raw.cache().prefetch(AUTOTUNE)
 # -------------------------------------------
 # 4) (Optional) Visual sanity check of images
 # -------------------------------------------
 def show_batch(ds, class_names, n=9, title="Sample images"):
 images, labels = next(iter(ds))
 n = min(n, images.shape[0])
 cols = int(math.sqrt(n))
 rows = math.ceil(n/cols)
 plt.figure(figsize=(2.8*cols, 2.8*rows))
 for i in range(n):
 plt.subplot(rows, cols, i+1)
 plt.imshow(images[i].numpy().astype("uint8"))
 plt.title(class_names[int(labels[i])])
 plt.axis("off")
 plt.suptitle(title)
 plt.tight_layout(); plt.show()
 show_batch(train_ds, class_names, n=9, title="Animal-6 — Training samples")
 # --------------------------------------------
 # 5) Build a SIMPLIFIED CNN (meets P2 intent)
✅

  # --------------------------------------------
 # Simplifications vs. a "typical" class model:
 # • Only 2 convolution blocks
 # • Small filter counts (16 → 32)
 # • Single small Dense layer (64)
 # • No heavy regularization yet (we'll do that in Problem 3 to push >85%)
 model = keras.Sequential([
 layers.Input(shape=IMG_SIZE + (3,)), # (H,W,C) with RGB images
 layers.Rescaling(1./255), # normalize pixels to [0,1]
 ])
layers.MaxPooling2D(),
 layers.Conv2D(32, 3, activation="relu"),
 layers.MaxPooling2D(),
 layers.Flatten(),
  # -------------------------------
 # 6) Compile and train the model
 # -------------------------------
 # Loss = sparse_categorical_crossentropy because labels are integer-encoded
 model.compile(
 optimizer=keras.optimizers.Adam(1e-3),
 loss="sparse_categorical_crossentropy",
 metrics=["accuracy"]
)
layers.Conv2D(16, 3, activation="relu"),
 layers.Dense(64, activation="relu"),
 layers.Dense(num_classes, activation="softmax"), # 6 outputs
 print("\nModel summary (simplified):")
 model.summary()
 EPOCHS = 12 # short baseline run; increase if underfitting
 history = model.fit(
 train_ds,
 validation_data=val_ds,
 epochs=EPOCHS

 )
 # ---------------------------------
 # 7) Plot training/validation curves
 # ---------------------------------
 plt.figure()
 plt.plot(history.history["accuracy"], label="train acc")
 plt.plot(history.history["val_accuracy"], label="val acc")
 plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Animal-6 (Simplified) — Accuracy")
 plt.legend(); plt.show()
 plt.figure()
 plt.plot(history.history["loss"], label="train loss")
 plt.plot(history.history["val_loss"], label="val loss")
 plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Animal-6 (Simplified) — Loss")
 plt.legend(); plt.show()
 # ---------------------------------
 # 8) Evaluate (val set as test proxy)
 # ---------------------------------
 test_loss, test_acc = model.evaluate(test_ds, verbose=0)
 print(f"\nTest (validation proxy) accuracy: {test_acc:.3f}")
 # ---------------------------------
 # 9) Confusion matrix (optional but great for your report)
 # ---------------------------------
 # Note: we need the raw (unprefetched) dataset to extract labels in order.
 # We stored it as 'test_ds_raw' before caching/prefetching.
 from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
 y_true = np.concatenate([y.numpy() for _, y in test_ds_raw])
 y_pred = np.argmax(model.predict(test_ds), axis=1)
 cm = confusion_matrix(y_true, y_pred)
 disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
 fig, ax = plt.subplots(figsize=(6,6))
 disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
 plt.title("Animal-6 — Confusion Matrix (Simplified)")
 plt.tight_layout(); plt.show()

 TensorFlow: 2.19.0
 Python: 3.12.12 (main, Oct 10 2025, 08:52:57) [GCC 11.4.0]
 Mounted at /content/drive
 Using dataset at: /content/drive/MyDrive/EE267/Animal_Dataset
/content/drive/MyDrive/EE267/Animal_Dataset -> ['cat', 'chicken', 'butterfly', 'elephant', 'horse', 'spider']
 Found 15912 files belonging to 6 classes.
 Using 12730 files for training.
 Found 15912 files belonging to 6 classes.
 Using 3182 files for validation.
 Classes: ['butterfly', 'cat', 'chicken', 'elephant', 'horse', 'spider']
 