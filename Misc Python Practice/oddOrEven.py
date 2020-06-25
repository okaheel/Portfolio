#Ask the user for a number. Depending on whether the number is even or odd, print out an appropriate message to the user. 
#Hint: how does an even / odd number react differently when divided by 2?
#If the number is a multiple of 4, print out a different message.
#Ask the user for two numbers: one number to check (call it num) and one number to divide by (check). If check divides evenly into num, tell that to the user. If not, print a different appropriate message.

inputNumber = int(input("Enter a number to check if Even or Odd: "))
check = int(input("Enter a number to check if the first number is a multiple of: "))
state = inputNumber%2
state4 = inputNumber%4

if(state == 0):
    if(state4 == 0):
        print("The number " + str(inputNumber )+ " is Even and a multiple of 4!")
    else:
        print("The number " + str(inputNumber) + " is Even and not a multiple of 4!")
else:
   print("The number " + str(inputNumber) + " is Odd!")