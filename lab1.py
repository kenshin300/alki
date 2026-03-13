#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Лабораторная работа "Введение в нейронные сети"
Выполнение всех практических заданий в одном скрипте.
Запуск: python lab_neural_networks.py
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Подавление предупреждений TensorFlow (опционально)
tf.get_logger().setLevel('ERROR')

print("="*60)
print("Лабораторная работа: Введение в нейронные сети")
print("="*60)

# ----------------------------------------------------------------------
# Часть 1.1: Тензоры, broadcasting
# ----------------------------------------------------------------------
print("\n[Часть 1.1] Тензоры и broadcasting")
scalar = tf.constant(5)
vector = tf.constant([1., 2., 3.])
matrix = tf.constant([[1., 2.], [3., 4.]])
tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float32)

print("Скаляр:", scalar.numpy(), "shape:", scalar.shape, "dtype:", scalar.dtype)
print("Вектор:", vector.numpy(), "shape:", vector.shape, "dtype:", vector.dtype)
print("Матрица:\n", matrix.numpy(), "shape:", matrix.shape, "dtype:", matrix.dtype)
print("3D тензор:\n", tensor_3d.numpy(), "shape:", tensor_3d.shape, "dtype:", tensor_3d.dtype)

a = tf.constant([1., 2., 3.])
b = tf.constant([[10.], [20.], [30.]])
c = a + b
d = a * b
print("\na:", a.shape, a.dtype)
print("b:", b.shape, b.dtype)
print("c (a + b):\n", c.numpy(), "shape:", c.shape)
print("d (a * b):\n", d.numpy(), "shape:", d.shape)
print("\nНажмите Enter, чтобы продолжить...")
input()

# ----------------------------------------------------------------------
# Часть 1.2: Линейная регрессия с tf.GradientTape
# ----------------------------------------------------------------------
print("\n[Часть 1.2] Линейная регрессия с GradientTape")
# Генерация данных: y = 3x - 2 + шум
np.random.seed(42)
x_data = np.linspace(-1, 1, 200, dtype=np.float32)
y_data = 3.0 * x_data - 2.0 + np.random.normal(scale=0.2, size=x_data.shape).astype(np.float32)

x = tf.constant(x_data)
y = tf.constant(y_data)

w = tf.Variable(0.0, dtype=tf.float32)
b = tf.Variable(0.0, dtype=tf.float32)
lr = 0.1
losses = []

for step in range(300):
    with tf.GradientTape() as tape:
        y_pred = w * x + b
        loss = tf.reduce_mean((y - y_pred) ** 2)
    dw, db = tape.gradient(loss, [w, b])
    w.assign_sub(lr * dw)
    b.assign_sub(lr * db)
    losses.append(loss.numpy())

print(f"Итоговые параметры: w = {w.numpy():.4f}, b = {b.numpy():.4f}")
print(f"Финальная loss = {losses[-1]:.6f}")

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(x_data, y_data, 'o', label='Исходные данные', markersize=3)
plt.plot(x_data, w.numpy() * x_data + b.numpy(), 'r-', label='Линейная регрессия')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Данные и восстановленная прямая')

plt.subplot(1,2,2)
plt.plot(losses)
plt.xlabel('Шаг')
plt.ylabel('MSE')
plt.title('Динамика ошибки')
plt.grid(True)
plt.show()

print("\nНажмите Enter, чтобы продолжить...")
input()

# ----------------------------------------------------------------------
# Часть 2: Персептрон для классификации Fashion-MNIST
# ----------------------------------------------------------------------
print("\n[Часть 2] Персептрон на Fashion-MNIST")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    validation_split=0.1,
                    epochs=10,
                    batch_size=128,
                    verbose=2)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss per epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy per epoch')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

print("\nНажмите Enter, чтобы продолжить...")
input()

# ----------------------------------------------------------------------
# Часть 3.1: Сравнение оптимизаторов
# ----------------------------------------------------------------------
print("\n[Часть 3.1] Сравнение оптимизаторов")

def train_perceptron(optimizer, name, epochs=10, batch_size=128):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    start = time.perf_counter()
    history = model.fit(x_train, y_train,
                        validation_split=0.1,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0)
    end = time.perf_counter()

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return {
        'name': name,
        'test_acc': test_acc,
        'time': end - start,
        'history': history
    }

optimizers = [
    (tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9), 'SGD+Momentum'),
    (tf.keras.optimizers.Adam(learning_rate=1e-3), 'Adam'),
    (tf.keras.optimizers.RMSprop(learning_rate=1e-3), 'RMSprop')
]

results_opt = []
for opt, name in optimizers:
    print(f"Обучение с {name}...")
    res = train_perceptron(opt, name)
    results_opt.append(res)

print("\n" + "="*50)
print(f"{'Optimizer':<15} {'Test Acc':<10} {'Time (s)':<10}")
print("-"*50)
for r in results_opt:
    print(f"{r['name']:<15} {r['test_acc']:.4f}     {r['time']:.2f}")
print("="*50)

plt.figure(figsize=(10,5))
for r in results_opt:
    plt.plot(r['history'].history['val_loss'], label=r['name'])
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Сравнение оптимизаторов по валидационной ошибке')
plt.legend()
plt.grid(True)
plt.show()

print("\nНажмите Enter, чтобы продолжить...")
input()

# ----------------------------------------------------------------------
# Часть 3.2: Сравнение batch size
# ----------------------------------------------------------------------
print("\n[Часть 3.2] Сравнение batch size")

batch_sizes = [32, 128, 256]
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
results_bs = []

for bs in batch_sizes:
    print(f"Batch size = {bs}")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    start = time.perf_counter()
    history = model.fit(x_train, y_train,
                        validation_split=0.1,
                        epochs=10,
                        batch_size=bs,
                        verbose=0)
    end = time.perf_counter()
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    results_bs.append({
        'batch_size': bs,
        'test_acc': test_acc,
        'time': end - start,
        'history': history
    })

print("\n" + "="*50)
print(f"{'Batch size':<12} {'Test Acc':<10} {'Time (s)':<10}")
print("-"*50)
for r in results_bs:
    print(f"{r['batch_size']:<12} {r['test_acc']:.4f}     {r['time']:.2f}")
print("="*50)

plt.figure(figsize=(10,5))
for r in results_bs:
    plt.plot(r['history'].history['val_accuracy'], label=f'batch={r["batch_size"]}')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Влияние batch size на сходимость')
plt.legend()
plt.grid(True)
plt.show()

print("\nНажмите Enter, чтобы продолжить...")
input()

# ----------------------------------------------------------------------
# Часть 4.1: Сохранение и загрузка модели
# ----------------------------------------------------------------------
print("\n[Часть 4.1] Сохранение и загрузка модели")

# Обучим модель заново для чистоты (можно взять лучшую из предыдущих)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=0)

# Сохраняем
model.save('fashion_perceptron.keras')
print("Модель сохранена как fashion_perceptron.keras")

# Загружаем
loaded_model = tf.keras.models.load_model('fashion_perceptron.keras')
print("Модель загружена.")

# Предсказание на 10 случайных тестовых изображениях
indices = np.random.choice(len(x_test), 10, replace=False)
x_sample = x_test[indices]
y_true = y_test[indices]

predictions = loaded_model.predict(x_sample, verbose=0)
pred_classes = np.argmax(predictions, axis=1)
pred_probs = np.max(predictions, axis=1)

print("\nИстинная метка | Предсказанный класс | Вероятность")
print("-" * 50)
for true, pred, prob in zip(y_true, pred_classes, pred_probs):
    print(f"       {true}               {pred}                {prob:.4f}")

# ----------------------------------------------------------------------
# Часть 4.2 (опционально): экспорт в TensorFlow Lite
# ----------------------------------------------------------------------
print("\n[Часть 4.2] Экспорт в TFLite")
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
tflite_model = converter.convert()
with open('fashion_perceptron.tflite', 'wb') as f:
    f.write(tflite_model)

size_keras = os.path.getsize('fashion_perceptron.keras') / 1024
size_tflite = os.path.getsize('fashion_perceptron.tflite') / 1024
print(f"Размер .keras: {size_keras:.2f} KB")
print(f"Размер .tflite: {size_tflite:.2f} KB")

# Проверка TFLite модели
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nПроверка TFLite на тех же 10 примерах:")
for i, idx in enumerate(indices):
    input_data = np.expand_dims(x_test[idx].astype(np.float32), axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    pred_class = np.argmax(output)
    prob = np.max(output)
    print(f"Пример {i+1}: истинная {y_test[idx]}, предсказанная TFLite {pred_class}, вероятность {prob:.4f}")

print("\nНажмите Enter, чтобы продолжить...")
input()

# ----------------------------------------------------------------------
# Часть 5.1: Реализация Back Propagation для XOR (batch)
# ----------------------------------------------------------------------
print("\n[Часть 5.1] Реализация Back Propagation для XOR (batch)")

# Данные XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

np.random.seed(42)
input_size = 2
hidden_size = 4
output_size = 1

W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))

lr = 0.5
epochs = 5000
loss_batch = []

for epoch in range(epochs):
    # Forward
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    loss = 0.5 * np.sum((y - A2) ** 2)
    loss_batch.append(loss)

    # Backward
    dZ2 = (A2 - y) * sigmoid_deriv(Z2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_deriv(Z1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

print("Предсказания после обучения (batch):")
A2_final = sigmoid(np.dot(sigmoid(np.dot(X, W1) + b1), W2) + b2)
for i in range(4):
    print(f"{X[i]} -> {A2_final[i,0]:.4f} (ожидалось {y[i,0]})")

plt.figure()
plt.plot(loss_batch)
plt.xlabel('Эпоха')
plt.ylabel('Loss (MSE)')
plt.title('Обучение на XOR (batch)')
plt.grid(True)
plt.show()

# ----------------------------------------------------------------------
# Часть 5.2: On-line обучение для XOR и сравнение
# ----------------------------------------------------------------------
print("\n[Часть 5.2] On-line обучение для XOR и сравнение с batch")

def train_xor_online(epochs=5000, lr=0.5):
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 0.1
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.1
    b2 = np.zeros((1, output_size))
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(X.shape[0]):
            x_i = X[i:i+1]
            y_i = y[i:i+1]

            # Forward
            Z1 = np.dot(x_i, W1) + b1
            A1 = sigmoid(Z1)
            Z2 = np.dot(A1, W2) + b2
            A2 = sigmoid(Z2)

            epoch_loss += 0.5 * (y_i - A2) ** 2

            # Backward
            dZ2 = (A2 - y_i) * sigmoid_deriv(Z2)
            dW2 = np.dot(A1.T, dZ2)
            db2 = np.sum(dZ2, axis=0, keepdims=True)

            dA1 = np.dot(dZ2, W2.T)
            dZ1 = dA1 * sigmoid_deriv(Z1)
            dW1 = np.dot(x_i.T, dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)

            # Update
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

        loss_history.append(epoch_loss / X.shape[0])
    return loss_history, W1, b1, W2, b2

loss_online, W1o, b1o, W2o, b2o = train_xor_online(epochs=5000)

# Точность batch
A2_batch = sigmoid(np.dot(sigmoid(np.dot(X, W1) + b1), W2) + b2)
pred_batch = (A2_batch > 0.5).astype(int)
acc_batch = np.mean(pred_batch == y)

# Точность on-line
A2_online = sigmoid(np.dot(sigmoid(np.dot(X, W1o) + b1o), W2o) + b2o)
pred_online = (A2_online > 0.5).astype(int)
acc_online = np.mean(pred_online == y)

print(f"Batch точность: {acc_batch*100:.1f}%")
print(f"On-line точность: {acc_online*100:.1f}%")

plt.figure()
plt.plot(loss_batch, label='batch')
plt.plot(loss_online, label='on-line')
plt.xlabel('Эпоха')
plt.ylabel('Средняя ошибка')
plt.title('Сравнение batch и on-line обучения на XOR')
plt.legend()
plt.grid(True)
plt.show()

print("\nВсе задания выполнены!")