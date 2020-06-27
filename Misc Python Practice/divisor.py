# coding: utf-8
#Create a program that asks the user for a number and then prints out a list of all the divisors of that number. 
# If you don’t know what a divisor is, it is a number that divides evenly into another number. 
# For example, 13 is a divisor of 26 because 26 / 13 has no remainder.)

def userPrompt():
    number = input("Input a number to get the divisors: ")
    return number

def findDivisors(large_num):
    results = []
    full_list = range(2, large_num)
    for number in full_list:
        if large_num%number == 0:
            results.append(number)
    return results

number = userPrompt()
divisors = findDivisors(number)
print(divisors)


