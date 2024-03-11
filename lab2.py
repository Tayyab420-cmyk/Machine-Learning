# try:
#     x=int(input("Enter a Number "))
#     y=10/x
#     print("Result",y)
# except ValueError:
#     print("Invalid Input")
# except ZeroDivisionError:
#     print("Cannot divide by zero")

# with open("sample.txt","w") as file:
#     file.write("Hello world!\.\n")
#     file.write("This is a sample file!\n")
#     file.write("Tayyab\n")
#     file.write("This is a sample file\n")
#     file.write("Tayyab Ahmad\n")

# with open("sample.txt","r") as file:
#         contents=file.read()
#         print("File contents",contents)

# with open("sample.txt","r") as file:
#     lines=file.readlines()
#     print("File contents as list:")
#     print(lines)
    

# def add(a,b):
#     return a+b
# def factorial(n):
#     if n ==0:
#       return 1
#     else:
#       return n*factorial(n-1)

# numbers=[1,2,3,4,5]
# doubled_numbers=list(map(lambda x : x * 2 , numbers))
# print("Doubled Numbers ", doubled_numbers)

# even_numbers=list(filter(lambda x: x % 2 == 0 , numbers))
# print("Even Numbers ", even_numbers)

# squares=[x**2 for x in numbers]
# print("saquares",squares)
   
# class person:
#     def __init__(self,name,age):
#         self.name = name
#         self.age = age
        
#         person1=person("Alice",30)
        
# class dog:
#     def __init__(self,name,breed):
#         self.name = name
#         self.breed = breed
        
#         def bark(self):
#             return  f"{self.name} says woof!"
        
#         dog1=dog("buddy")
#         print(dog1.bark())
    
# class animal:
#     def speak(self):
#         raise NotImplementedError("Subclasses must implement this method.")
    
# class Dog(animal):
#     def speak(self):
#         return "woof!"
    
# from abc import ABC,abstractmethod
# class Animal(ABC):
#     def move(slef):
#         pass
#     class car(vehicle):
#         def move(self):
#             return "The car moves."


# class A:
#             def process(self):
#                 return "A"
#             class B(A):
#                 def process(self):
#                     return "B"
#                 class C(A):
#                     def process(self):
#                         return "c"
#                     class D(B, C):
#                         pass
# print(D().process())

#.................................. Task 1............................
# with open ('task.txt', 'w') as file:
#     file.write('Hello World! \nThis is a test of the emergency broadcast system.\n' )
#     file.write("My name is khan And I am not a terrorist")
# with open('task.txt','r') as file:
#     contents=file.read()
#     print("File contents",contents)
    
    
#....................................Task 2............................
def read_file(file_name):
    
    with open("sample.txt", 'r') as f:
        file_contents = f.read()
        print(file_contents)
        return file_contents

def read_file_into_list(file_name):
 
    with open(file_name, 'r') as f:
        file_list = f.readlines()
        return file_list

def write_first_line_to_file(file_contents, output_filename):

    first_line = file_contents.split('\n')[0]
    with open(output_filename, 'w') as f:
        f.write(first_line)

def read_even_numbered_lines(file_name):

    with open(file_name, 'r') as f:
        file_list = f.readlines()
        even_numbered_lines = file_list[::2]
        return even_numbered_lines

def read_file_in_reverse(file_name):
  
    with open(file_name, 'r') as f:
        file_list = f.readlines()
        file_list.reverse()
        print(file_list)
        return file_list
def main():
    file_contents = read_file("sampletext.txt")
    print(read_file_into_list("sampletext.txt"))
    write_first_line_to_file(file_contents, "online.txt")

