import requests
from bs4 import BeautifulSoup
import numpy as np


# Task 1
class Employee:
    total_employees = 0
    salary_sum = 0

    def __init__(self, name, fam, sal, dept):
        self.name = name
        self.family = fam
        self.__salary = float(sal)
        self.__department = dept

        # Increment class data
        Employee.total_employees += 1
        Employee.salary_sum += self.__salary

    def set_salary(self, new_sal):
        # verify that the salary is a float
        new_sal = float(new_sal)

        # adjust the average salary for the new salary change
        Employee.salary_sum += (new_sal - self.__salary)
        self.__salary = float(new_sal)

        # return the employee object back to the caller
        return self

    def get_salary(self):
        return self.__salary

    def set_department(self, new_dept):
        self.__department = new_dept
        return self

    def get_department(self):
        return self.__department

    def average_salary(self):
        return self.__class__.salary_sum / self.__class__.total_employees


# Create a Fulltime Employee class that inherits from Employee
class FulltimeEmployee(Employee):
    def print_something(self):
        print("I inherited from Employee")


# ------------------------------------------MAIN-------------------------------------------------------
# Q1.E
emp = Employee('John', 'Smith', 20_000, 'Flooring')
print("average salary: " + str(emp.average_salary()))
full_time_emp = FulltimeEmployee('Ron', 'Swanson', 50_000, 'Management')
print("average salary: " + str(full_time_emp.average_salary()))
full_time_emp.print_something()

emp.set_salary(10_000)
print("salary of emp 1 after salary change: " + str(emp.get_salary()))
print("avg salary after emp 1 salary change: " + str(emp.average_salary()))

print("Number of Employees: " + str(Employee.total_employees))

#Q2
html = requests.get('https://en.wikipedia.org/wiki/Deep_learning')
parsed_html = BeautifulSoup(html.content, 'html.parser')
print("The title of the webpage is: " + parsed_html.title.string)

print("Here are all of the links on the wiki-page: \n")
links = parsed_html.find_all('a')
for link in links:
    print(link.get('href'))

# Q3
random_array = np.random.randint(20, size=15)
print("random array: ")
print(random_array)
new_array = random_array.reshape(3,5)
print("reshaped array: ")
print(new_array)
max_rows = new_array.max(axis=1, keepdims=True)
replaced_maxes = np.where(new_array != max_rows, new_array, 0)
print("With maxes replaced: ")
print(replaced_maxes)