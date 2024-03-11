from abc import ABC, abstractmethod

class Bank(ABC):
    def basicinfo(self):
        print("This is a generic bank")
        return "Generic bank: 0"

    @abstractmethod
    def withdraw(self, amount):
        pass

class Swiss(Bank):
    def __init__(self):
        self.bal = 1000

    def basicinfo(self):
        print("This is the Swiss Bank")
        return f"Swiss Bank: {self.bal}"

    def withdraw(self, amount):
        self.bal -= amount
        print(f"Withdrawn amount: {amount}")
        print(f"New balance: {self.bal}")
        return self.bal


swiss_bank = Swiss()
print(swiss_bank.basicinfo())  
                               

swiss_bank.withdraw(30)         
                                
