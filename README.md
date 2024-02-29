# Home_Work
Метод k ближайших соседей

# Проект классификации методом k-NN
Данный проект реализует алгоритм k-Nearest Neighbors (k-NN) для классификации и оценивает его производительность на наборе данных о раке груди.

## Файлы
- `distances.py`: Содержит функции для вычисления евклидового, манхэттенского и расстояния Чебышёва.
- `knn.py`: Реализует классификатор k-NN с методами fit и predict.
- `compute.py`: Загружает данные о раке груди, разделяет их на обучающий и тестовый наборы, выполняет классификацию методом k-NN и оценивает метрики производительности.

## Набор данных
Используется набор данных о раке груди [Breast Cancer dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset) из библиотеки scikit-learn. Он содержит количественные признаки, а целевая переменная является бинарной.

## Использование
1. Запустите `compute.py` для выполнения классификации и оценки.
2. Настройте параметры, такие как `random_state` и `k`, в скрипте для различных конфигураций.

## Результаты
Результаты для различных значений `random_state` и `k` следующие:

### Random State: 42

- k: 3
  - Accuracy: 0.92
  - Precision: 0.95
  - Recall: 0.89
  - F1 Score: 0.92

- k: 5
  - Accuracy: 0.94
  - Precision: 0.97
  - Recall: 0.91
  - F1 Score: 0.94

- k: 7
  - Accuracy: 0.93
  - Precision: 0.96
  - Recall: 0.91
  - F1 Score: 0.93

### Random State: 1

- k: 3
  - Accuracy: 0.91
  - Precision: 0.95
  - Recall: 0.87
  - F1 Score: 0.91

- k: 5
  - Accuracy: 0.94
  - Precision: 0.96
  - Recall: 0.93
  - F1 Score: 0.94

- k: 7
  - Accuracy: 0.93
  - Precision: 0.96
  - Recall: 0.91
  - F1 Score: 0.93
