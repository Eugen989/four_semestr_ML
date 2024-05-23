# http://localhost:5000/api?sepal_length=5.1&sepal_width=3.5&petal_length=1.4&petal_width=0.2

from requests import get
# height = input('Введите height = ')
# mark = input('Введите mark (1-5) = ')
# end_student = input('Введите end (0-1) = ')
# salary = input('Введите salary = ')

height = 185
mark = 5
end_student = 1
salary = 25000

response = get(f'http://localhost:5000/api?height={height}&mark={mark}&end={end_student}&salary={salary}')
response.raise_for_status()  # Проверка, не произошла ли HTTP ошибка
data = response.json()       # Попытка декодировать JSON ответ
print(data)