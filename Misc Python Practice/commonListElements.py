#write a program that returns a list that contains only the elements that are common between the lists (without duplicates). 
# Make sure your program works on two lists of different sizes.

a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def findCommonElements(list1, list2):
    commonElements = []
    for element in list1:
        if element in list2:
            if element not in commonElements:
                commonElements.append(element)
    
    return commonElements

commonNums = findCommonElements(a,b)
print(commonNums)

    
                
        
