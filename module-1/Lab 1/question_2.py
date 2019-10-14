# 2) soln:
# Create a dictionary with keys as names and values as list of (subjects, marks) in sorted order.

student_list = [
    ('John', ('Physics', 80)) ,
    ('Daniel', ('Science', 90)),
    ('John', ('Science', 95)),
    ('Mark',('Maths', 100)),
    ('Daniel', ('History', 75)),
    ('Mark', ('Social', 95))
]

student_dict = dict()

for student in student_list:
    if student[0] not in student_dict:
        student_dict[student[0]] = list()

    student_dict[student[0]].append(student[1])

for student in student_dict:
    print(student, ":", sorted(student_dict[student]))
