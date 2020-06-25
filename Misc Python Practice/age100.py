#Create a program that asks the user to enter their name and their age. 
#Print out a message addressed to them that tells them the year that they will turn 100 years old.
#Add on to the previous program by asking the user for another number and printing out that many copies of the previous message. (Hint: order of operations exists in Python)
#Print out that many copies of the previous message on separate lines. (Hint: the string "\n is the same as pressing the ENTER button)
import datetime

name = input("What is your name? ")
age = input("what is your age? ")
number_prints = input("How many times would you like to print it? ")

yearWhen100 = datetime.datetime.now().year - int(age) + 100
stringToPrint = name + ' will be 100 in the year ' + str(yearWhen100)

for i in range(int(number_prints)):
    print(stringToPrint)