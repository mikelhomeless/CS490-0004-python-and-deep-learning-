# # 1) soln:
# temp = " "
# dict = {}
# for j in range(len(str)):
#    for i in range (j, len(str)):
#        if not (str[i] in temp):
#           temp += str[i]
#       else:
#       dict[temp] = len(temp)
#       temp = ''
#       break
# max_val = max(dict.values())
# list1=[]
# for key, val in dict.items();
#   if max_val == val:
#      list1.append((key, val))
#
#
#
# # 2) soln:
# # Create a dictionary with keys as names and values as list of (subjects, marks) in sorted order.
# student_list = {
#     John :   [("Physics", 80), ("Science", 95)],
#     Daniel:  [ ("History", 75), ("Science", 90)],
#     Mark:    [ ("Maths", 100), ("Social", 95)]
# }
#
# for student in student_list:
#   if student[0] not in student_dict:
#      student_dict[student[0]] = list()
#      student_dict[student[0]].append(student[1])
#   else:
#      student_dict[student[0]].append(student[1])
#
# for student in student_dict;
#     print(student, ":", sorted(student_dict[student]))
#
# # 3)
# # soln:
# # > I have created the Library management system, which has 5 different
# #   classes namely Person, Student, Librarian, Book, Borrow_book.
# # > Person is the main class and the classes Student and Librarian
# #   have inherited (single inheritance) the Person class.
# # > Borrow_Book class implements multiple inheritance with base class
# #   Student, Book.
# # > Declared private data member StudentCount in student class to count
# #   number of student objects created.
# # > Used super call in class Librarian to initialize Person class object.
# # > Used a private member __numBooks for keeping the track of books.
#
# class Person:
#   def__init__(self, name, email):
#      self.name = name
#      self.email = email
#
#   def display (self):
#      print("Name: ", self.name)
#      print("Email: ", self.email)
#
# #Inheritance concept where student, is inheriting the person class
# class Student(Person);
#   StudentCount = 0
#
#  def__init__(self, name, email, student_id):
#      Person.__init__(self, name, email)
#      self.student_id = student_id
#      student.StudentCount +=1
#
# #super class in the librarian class
# class Librarian(Person);
#   StudentCount = 0
#
#  def__init__(self, name, email, employee_id):
# #super call where librarian class is inheriting the Person class
#      super().__init__(name, email)
#      self.employee_id = employee_id
#
# #creating a private number
# __numBooks = 0 #private number
#   def__init__(self, book_name, auhtor, book_id):
#         self.book_name = book_name
#         self .author = author
#         self.book_id = book_id
#         Book.__numBooks +=1 #keeps track of which student or staff has book checked
#
# #multiple inheritance concept for Borrow_Book class
# class Borrow_Book(Student, Book)
#     def__init__(self,name, email, student_id, book_name, author, book_id):
#         Student.__init__(self, name, email, student_id)
#         Book.__init__(self, book_name, author, book_id)
#
#     def diplay(self):
#         print("Borrowed Book Details:")
#         student.display(self)
#         Book.display(self)
#
# #creating instance of all classes
# Records = []
# Records.append(Student('Anisha', 'anishax01@gmail.com', 123)
# Records.append(Librarian('Michael', 'michaely02@gmail.com', 456)
# Records.append(Book('Notebook', 'Syed', 789)
# Records.append(Borrow_Book('Saira', 'sairaz03@gmail.com', 101, 'Fault in our Stars', 'John Green', 987650))
