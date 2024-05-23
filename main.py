import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Шаг 1: Генерация точек, напоминающих окружность
num_points = 10000
theta = np.linspace(0, 2 * np.pi, num_points)
r = np.sqrt(np.random.rand(num_points))
x1 = r * np.cos(theta)
y1 = r * np.sin(theta)
class_1 = np.ones(num_points)

# Шаг 2: Генерация второго ряда точек с другим центром
x_shift, y_shift = 1.5, 1.5
x2 = r * np.cos(theta) + x_shift
y2 = r * np.sin(theta) + y_shift
class_2 = np.full(num_points, 2)

# Шаг 3: Сдвиг второго ряда точек (уже сделано в шаге 2)

# Шаг 4: Объединение и перемешивание точек
x = np.concatenate((x1, x2))
y = np.concatenate((y1, y2))
classes = np.concatenate((class_1, class_2))

data = pd.DataFrame({'x': x, 'y': y, 'class': classes})
data = data.sample(frac=1).reset_index(drop=True)  # Перемешиваем данные

# Визуализация данных
plt.scatter(data['x'], data['y'], c=data['class'], cmap='viridis', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Сгенерированные точки')
plt.show()

# Шаг 5: Разделение на тренировочную и тестовую выборку
X = data[['x', 'y']]
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Шаг 6: Обучение модели и проверка её работы
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Визуализация результатов предсказания
plt.scatter(X_test['x'], X_test['y'], c=y_pred, cmap='viridis', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Результаты предсказания')
plt.show()
