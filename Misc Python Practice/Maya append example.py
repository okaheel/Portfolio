# animals list
animals = ['cat', 'dog', 'rabbit']

# 'guinea pig' is appended to the animals list
temp2 = animals[1]
animals[1] = 'guinea pig'
animals.append(temp2)

# Updated animals list
print('Updated animals list: ', animals)