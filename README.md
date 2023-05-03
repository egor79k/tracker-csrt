# Алгоритмы автоматической вертикальной сшивки изображений компьютерной томографии

Инструмент для сшивки трехмерных томографических реконструкций предоставляет пользовательский интерфейс со следующим функционалом:
-	Открытие реконструкций в нескольких форматах: набор изображений аксиальных срезов, конфигурационный файл одной реконструкции, конфигурационный файл набора реконструкций.
-	Управление набором сшиваемых реконструкций в виде списка с возможностью добавления, удаления и изменения порядка.
-	Выбор одного из реализованных способов сшивки: без сшивки (для совместной демонстрации сшиваемых реконструкций), прямое выравнивание, SIFT, адаптированный SIFT.
-	Визуализация срезов сшитой реконструкции с возможностью выбора одной из трех плоскостей (сагиттальная, фронтальная, аксиальная) и номера слоя.
-	Сохранение полученного результата в одном из поддерживаемых форматов.
- Экспорт изображений срезов.

### Зависимости:
- [OpenCV](https://opencv.org/)
- [Qt5](https://doc.qt.io/qt-5/)
- [TinyTIFF](https://github.com/jkriege2/TinyTIFF)

### Сборка:
```
cd vertical_stitching
mkdir build
cd build
cmake ..
make
```

### Использование:
Поддерживается загрузка реконструкций в виде произвольного набора монохромных TIFF изображений одинакового размера, но удобнее использовать конфигурационный JSON файл [специального формата](https://docs.google.com/document/d/1qpn4UVwJcOSLZnA9c5tmEAUKEiq059hcWFhZNmENg-g/edit?usp=sharing). С его помощью можно загружать как одну цельную реконструкцию, так и несколько частей одновременно. Несколько демонстрационных реконструкций можно скачать [здесь](https://disk.yandex.ru/d/m8-yI4MHTXK8GQ). Реконструкции можно генерировать с помощью генератора [stacked-tomoscan-gen](https://github.com/egor79k/stacked-tomoscan-gen), созданного специально для тестирования алгоритмов вертикальной сшивки.

Также есть возможность отдельного использования библиотеки с алгоритмами в коде. Как подключить:
- Скопировать директорию `vertical_stitching/src/stitcher в проект,
- Добавить в CMakeLists проекта:
```
include_directories(stitcher)
...
target_link_libraries(<your_project> stitcher)
```
- Загрузка, сшивка и сохранение реконструкции:
```cpp
#include <sift_2d_stitcher.h>

VoxelContainer part_1, part_2;
SIFT2DStitcher stitcher;

// Load reconstruction parts using info files
part_1.loadFromJson('rec_part_1/info.json');
part_2.loadFromJson('rec_part_1/info.json');

// Stitching
std::shared_ptr<VoxelContainer> result = stitcher.stitch(part_1, part_2);

// Saving stitched reconstruction
result->saveToJson('rec_stitched');
```

## Ссылки:
- [Original article](https://arxiv.org/pdf/1611.08461v2.pdf)
