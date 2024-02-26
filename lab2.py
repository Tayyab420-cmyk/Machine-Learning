# try:
#     x=int(input("Enter a Number "))
#     y=10/x
#     print("Result",y)
# except ValueError:
#     print("Invalid Input")
# except ZeroDivisionError:
#     print("Cannot divide by zero")

# with open("sample.txt","w") as file:
#     file.write("Hello world!")
#     file.write("This is a sample file!")

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
   
class person:
    def __init__(self,name,age):
        self.name = name
        self.age = age
        
        person1=person("Alice",30)
        
class dog:
    def __init__(self,name,breed):
        self.name = name
        self.breed = breed
        
        def bark(self):
            return  f"{self.name} says woof!"
        
        dog1=dog("buddy")
        print(dog1.bark())
    
class animal:
    def speak(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
class Dog(animal):
    def speak(self):
        return "woof!"
    
        