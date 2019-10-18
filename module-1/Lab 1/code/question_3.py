# 3)
# soln:
# > I have created the Library management system, which has 5 different
#   classes namely Person, Student, Librarian, Book, Borrow_book.
# > Person is the main class and the classes Student and Librarian
#   have inherited (single inheritance) the Person class.
# > Borrow_Book class implements multiple inheritance with base class
#   Student, Book.
# > Declared private data member StudentCount in student class to count
#   number of student objects created.
# > Used super call in class Librarian to initialize Person class object.
# > Used a private member __numBooks for keeping the track of books.


class Person:
    def __init__(self, name, email):
     self.name = name
     self.email = email

    def display(self):
        print("Name: ", self.name)
        print("Email: ", self.email)


#Inheritance concept where student, is inheriting the person class
class Student(Person):
    StudentCount = 0

    def __init__(self, name, email, student_id):
        Person.__init__(self, name, email)
        self.student_id = student_id
        Student.StudentCount += 1


#super class in the librarian class
class Librarian(Person):
    StudentCount = 0

    def __init__(self, name, email, employee_id):
        #super call where librarian class is inheriting the Person class
        super().__init__(name, email)
        self.employee_id = employee_id


class Book:
    #creating a private number
    num_books = 0 #private number

    def __init__(self, book_name, author, book_id):
        self.book_name = book_name
        self.author = author
        self.book_id = book_id
        Book.num_books +=1 #keeps track of which student or staff has book checked

    def display(self):
        print('Author: {}, Title: {}, ID {}'.format(self.author, self.book_name, self.book_id))


#multiple inheritance concept for Borrow_Book class
class Borrow_Book:
    def __init__(self, student, book):
        self.student = student
        self.book = book

    def display(self):
        print("Borrowed Book Details:")
        self.student.display()
        self.book.display()

#creating instance of all classes
students = []
books = []
employees = []
rentals = []

students.append(Student('Anisha', 'anishax01@gmail.com', 123))
employees.append(Librarian('Michael', 'michaely02@gmail.com', 456))
books.append(Book('Notebook', 'Syed', 789))

student = students[0]
rentals.append(Borrow_Book(student, books[0]))

print('Employees: ', end='\n\n')
employees[0].display()

print('Current rental: ', end='\n\n')
rentals[0].display()
