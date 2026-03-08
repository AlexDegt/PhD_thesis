import os

# Указываем имена файлов
train_file = "aclr_correct_2_dim_train.txt"
test_file = "aclr_correct_2_dim_test.txt"
output_file = "aclr_correct_2_dim.txt"

# Работаем в текущей директории
with open(train_file, 'r') as f_train, \
     open(test_file, 'r') as f_test, \
     open(output_file, 'w') as f_out:
    
    # Читаем все строки из обоих файлов
    train_lines = f_train.readlines()
    test_lines = f_test.readlines()
    
    # Определяем максимальную длину
    max_lines = max(len(train_lines), len(test_lines))
    
    # Записываем строки поочередно
    for i in range(max_lines):
        # Если есть строка из train файла, записываем ее (нечетная строка)
        if i < len(train_lines):
            f_out.write(train_lines[i])
        else:
            f_out.write('\n')  # или можно пропустить
        
        # Если есть строка из test файла, записываем ее (четная строка)
        if i < len(test_lines):
            f_out.write(test_lines[i])
        else:
            f_out.write('\n')  # или можно пропустить

print(f"Файл {output_file} успешно создан в папке со скриптом!")