#rock rock -
#rock paper - 
#rock sissors -
#paper rock -
#paper paper - 
#paper sissors -
#sissors rock -
#sissors paper -
#sissors sissors -

from random import randint

moves = ["rock", "paper", "sissors"]

def ruleMachine(playerPick, ComputerPick):
    draw = "This round is draw!"
    win = "You won this round! The computer lost!"
    loss = "You lost this round! The computer won, better luck next time!"
    invalidInput = "Invalid input!"
    if(playerPick.lower() == ComputerPick.lower()):
        return(draw)
    elif(playerPick.lower() == 'rock' and ComputerPick.lower() == 'paper'):
        return(loss)
    elif(playerPick.lower() == 'rock' and ComputerPick.lower() == 'sissors'):
        return(win)
    elif(playerPick.lower() == 'paper' and ComputerPick.lower() == 'rock'):
        return(win)
    elif(playerPick.lower() == 'paper' and ComputerPick.lower() == 'sissors'):
        return(loss)
    elif(playerPick.lower() == 'sissors' and ComputerPick.lower() == 'rock'):
        return(loss)
    elif(playerPick.lower() == 'sissors' and ComputerPick.lower() == 'paper'):
        return(loss)
    else:
        return(invalidInput)

userMove = input("Welcome to Rock/Paper/Sissors. What is your move? ")
computerMove = moves[randint(0, 2)]

result = ruleMachine(userMove, computerMove)
print("Your move: " + userMove.lower() + " Computer Pick: " + computerMove)
print(result)


