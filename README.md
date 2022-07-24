# Нахождение экстремума функции
![условия задачи](https://github.com/stepyndriyy/tinyLBFGS/blob/main/problem.png)

## Краткий отчет
[Наблюдать мои попытки и улучшения можно тут.](analysis.ipynb)
Первой идеей стал градиентный спуск. (Замечание) Использовался критерий остановки - расстояние до точки ответа, но такой критерий можно использовать в исключительных случаях когда мы знаем ответ. После нескольких попыток и улучшений, используя adam я смог достичь требуемого количества итераций, однако на этом я не остановился. После изучения Ньютоновских и Квази-Ньютоновских методов стало ясно что их применение значительно улучшит результат. Но Метод Ньютона использует Гессиан, для его подсчета требуются вторые производные, использование которых запрещено по условию. Квази-Ньютоновские методы работают схожим образом, но не считаю гессиан явно, а лишь приближают его различными методами. Таким образом я пришел к BFGS и его улучшению L-BFGS. Этот метод имеет несколько преимуществ.
1. Сверхлинейная скорость сходимости(в отличие от линейной для градиентного спуска), и один из лучших показателей на практике.
2. сложность итерации O(n 2) для BFGS и O(n*m) для L-BFGS, где m "длина истории".
3. Обладает свойством самокоррекции.(Если условия нарушились, алгоритм не ломается а "восстанавливается" сам по себе).

При помощи этого алгоритма я достиг отличных результатов. Вызвав функции из "коробки" метод показал 80 итераций для данной по условию функции. Реализация L-BFGS Приведена на c++.

## Результаты
C++ реализация имеет простую структуру из .h и .cpp файла. компиляция не требует CMAKE или других систем сборки, достаточно вызвать одну команду(на Linux):
```bash
g++ main.cpp -o main
./main
```
Личный результат 109 итераций, при точности 1e-4. Текущая версия далека от идеала, существует множество оптимизаций для этого алгоритма описанных в книгах и реализованных в открытом доступе.

## Ссылки
**[1]** Jorge Nocedal Stephen J. Wright. Numerical Optimization (2006)