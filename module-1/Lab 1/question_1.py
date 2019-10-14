# 1) soln:
temp = " "
dictionary = {}
for j in range(len(str)):
    for i in range(j, len(str)):
        if not (str[i] in temp):
            temp += str[i]
        else:
            dictionary[temp] = len(temp)
            temp = ''
            break

max_val = max(dictionary.values())
list1 = []
for key, val in dictionary.items():
    if max_val == val:
        list1.append((key, val))
