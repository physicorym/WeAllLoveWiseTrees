# WeAllLoveWiseTrees
## Техническое решение
Техническое решение состоит в использовании классических алгоритмов компьютерного зрения. Разработанный алгоритм можно представить в 2 этапа:
### 1) Поиск битых пикселей.
Поиск битых пикселей осуществляется скользящим окном по картинке. Скользящее окно перемещается по каждому пикселю и высчитывает среднее значение окружающих его пикселей. В случае выброса значение центрального пикселя меняется на среднее.
Для обработки внешних пикселей кропы берутся с кастомным паддингом. Значение добавляемых пикселей также берется от среднего ближайших соседей.
### 2) Поиск координат. Поиск координат определяется с помощью выделения дескрипторов на анализируемых кропах и подложках. Для обеспечения быстродействия анализируемая подложка сжимается в два раза.
После опорные точки кропа сравниваются с точками подложки. Наиболее подходящее место отмечается координатами.

### Проведенные эксперименты
Ранее были протестированы методы обучения глубоких нейронных сетей (ResNet, Yolov и другие), однако в виду особенностей данных точность поиска координат по дескрипторам и опорным точка оказался выше.
Под особенностями данных данных можно выделить: значительное классовое разнообразие подложек, большое разрешение фотографий
## Как запустить
### Без использования Docker
#### Требования
- Python 3.9

#### Установка
В корневой директории запустить

```bash
python install -r requirements.txt
```

#### Запуск
1. Заполнить директорию layouts файлами подложек.
2. В корневой директории запустить

```python worker/main.py```

Инициирует запуск сервиса на порту 8000.
При необходимости, можно задать переменную окружения `APP_PORT`, в которой указывается номер порта.

### С использованием Docker
#### Установка
1. Заполнить директорию layouts файлами подложек.
2. В корневой директории запустить

```bash
docker build -t lctworker .
```

#### Запуск
```bash
docker container run -p 8000:8000 lctworker
```

## Взаимодействие с сервисом
Взаимодействие с сервисом предполагает два типа запросов.
1. Отправка POST-запроса на детекцию. Пример:
```python
import tifffile as tiff

image = tiff.imread(file_path)
height, width, _ = image.shape
payload = {'layout_name': layout_name, "crop_height": height, "crop_width": width}
files = {'file': ('filename', img.tobytes())}
r = requests.post('http://127.0.0.1:8000/', params=payload, files=files)
```

2. Получение результата детекции по GET-запросу с заданием `task_id`. Пример:
```python
r = requests.post('http://127.0.0.1:8000/', params=payload, files=files)
task_id = r.json()
r = requests.get('http://127.0.0.1:8000/', params=task_id, timeout=60)
print("Result: ", r.json())
```

## Как это работает
Клиент формирует и отправляет POST-запрос по URL `/`, указывая `layout_name` (название подложки), `crop_height` (высота
кропа) и `crop_width` (ширина кропа) в качестве payload и `file` (содержимое кропа).

Запрос приходит в `Worker.detect`, парсится в объект `Task` и помещается в очередь на детекцию.
Выделенная под задачу детекции корутина `process_task` ожидает задачу в очереди, принимает ее и создает новый процесс,
передавая задачу аргументом на обработку. Перед началом детекции засекается стартовое время, затем происходит
последовательная обработка детектором битых пикселей, детектором поиска координат кропа на подложке, фиксируется время
окончания детекции и помещение результатов детекции в оперативную память.

По GET-запросу по URL `/` с указанием `task_id` возвращаются результаты детекции. Если детекция еще в процессе, метод
обработки запроса будет дожидаться окончания детекции и только потом вернет результат.

![System design](docs/System%20Design-Page.png)
