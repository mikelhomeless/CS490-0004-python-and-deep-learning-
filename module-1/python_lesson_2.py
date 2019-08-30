LBS_TO_KG_CONVERSION_RATE = 0.453592


# Question 1
def lbs_to_kg():
    iterations = int(input("Enter the number of weights you would like to enter: "))
    weights_list = []
    for x in range(iterations):
        weights_list.append(int(input("Enter item %i : " % (x + 1))))

    # use a list comprehension to convert the weights into kg
    converted_weights = [weight * LBS_TO_KG_CONVERSION_RATE for weight in weights_list]
    print(converted_weights)


# Question 2
def string_alternative():
    input_string = input('Enter a string to alternate: ')
    print(input_string[::2])


# Question 3
def file_word_count():
    file_name = input('Enter the file name you would like to do a word-count on: ')
    words_dictionary = read_file(file_name)

    # write back to file
    with open(file_name, 'w') as file:
        for word in words_dictionary:
            file.write('%s: %i\n' % (word, words_dictionary[word]))


def read_file(file_name):
    words_dictionary = dict()
    file = open(file_name, 'r')
    for line in file:
        for word in line.split():
            if word in words_dictionary:
                words_dictionary[word] += 1
            else:
                words_dictionary[word] = 1

    file.close()
    return words_dictionary


lbs_to_kg()
string_alternative()
file_word_count()
