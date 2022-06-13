def index_relu():
    index_li = []
    for i in range(1, 14, 2):
        if i == 5 or i == 7:
            index_li.append(i+1)
        else:
            index_li.append(i)

    return index_li

# test
layer = []

for i in range(1, 11):
    if i in [1, 2, 4, 5, 7, 8]:
        layer.append('Conv')
    elif i in [3, 6, 9]:
        layer.append('pool')
    elif i == 10:
        for i in range(2):
            layer.append('affine')
            layer.append('Dropout')

# for i in index_relu():
#     if i == 9:
#         continue
#     else:
#         layer.insert(i, 'relu')