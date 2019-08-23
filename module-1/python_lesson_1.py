# QUESTION 1: What is the difference between python 2 and 3?
#   Answer: Some of the differences are as follows
#           Print Statement
#               - Python 2 allowed for a print statement to be called without using parentheses
#               _ Python 3 Forces parentheses to be used on the print statement
#
#           Division Operations
#               2/3 in python 2 will result in integer division => 1
#               2/3 in python 3 will result in a floating point division => 1.5
#
#           Input Function
#               The input method in python 2 is raw_input()
#               The input method in python 3 is simply input()

# QUESTION 2
user_string = input('Please type "python": ')
print(user_string[::2][::-1])

num1 = int(input('Please enter a number: '))
num2 = int(input('Please enter another number: '))
print('The sum of the numbers you entered is %s' % (num1 + num2))
print('The power of the first number by the second number is %s' % num1 ** num2)
print('The remainder of the first number divided by the second number is %s' % (num1 % num2))

# QUESTION 3
string_of_pythons = input('Please enter a string containing at least one "python": ')

# split the string on the word python and rejoin with pythons in the split locations
new_string = 'pythons'.join(string_of_pythons.split('python'))
print(new_string)