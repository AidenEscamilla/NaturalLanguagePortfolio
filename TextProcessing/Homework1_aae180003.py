import sys  # to get the system parameter
import os   # used by pathName
import re   #regex 
import pickle #for pickling object

#import pathlib # used by method 2


#This function is from class
def pathName(filepath):


    with open(os.path.join(os.getcwd(), filepath), 'r') as f:
        text_in = f.read()
    return text_in



def checkID(id):
    valid = False
    regex = '[A-Z]{2}\d{4}$'               #2 capital letters and 4 digits only
    inp = id
    if not re.match(regex, id):            #if its invalid
        while not valid:                   #loop until valid
            print('ID invalid:' + inp)
            print('ID is two Capital letters followed by 4 digits')
            inp = raw_input("Please enter a valid id: ")

            if re.match(regex, inp):        #check if its valid
               valid = True                 #break while
               
    return inp                              #return valid ID

def checkPhone(phoneNumber):
    valid = False
    regex = '^(1-)?\d{3}-\d{3}-\d{4}$'               #2 capital letters and 4 digits only
    inp = phoneNumber
    if not re.match(regex, phoneNumber):            #if its invalid
        while not valid:                   #loop until valid
            print('Phone ' + inp + ' is invalid')
            print('Enter phone number in form 123-456-7890')
            inp = raw_input("Enter phone number: ")

            if re.match(regex, inp):        #check if its valid
               valid = True                 #break while
               
    return inp                              #return valid ID

    


class Person:
    def __init__(self, name, ID, phoneNumber):
        self.name = name
        self.ID = ID
        self.phoneNumber = phoneNumber
        
    def display(self):
        print('\n' + self.name + ', ID: ' + self.ID + '\nPhone#: ' + self.phoneNumber)


def processFile(buf):
    buffer = str(buf)
    workers = {}

    buffer = buffer.replace("\n", "").split('\r')[1:]            #clean up by deleting newlines and throw away first line which is the document format and split by '\r'

    

    for data in buffer:
        data = data.split(',')                                  #split by commas
        name = [x.capitalize() for x in data[0:3]]              #capitalize names
        
        for index, part in enumerate(name):                     #Add "X" for no Middle Initial
            if not part:
                name[index] = 'X'

        name = name[0] +' '+ name[2] +' '+ name[1]                                   #Join name together in 1 string
        id = data[3].upper()                                    #assign ID 
        phoneNum = data[4]                                      #assign phone number

        id = checkID(id)                                        #check if id and phone number are valid with functions
        phoneNum = checkPhone(phoneNum)
       
        if not workers.get(id, False):                              #check if the id is repeated, works even if they changed an invalid id to one that already exists
            workers[id] = Person(name, id, phoneNum)                #Dictionary takes in id as a str, IF THAT BECOMES AN ISSUES look here
        else:
            print('Error: Employee not added, id repeat found')
            quit()
    return workers




if __name__ == '__main__':
    employees = None
  
    if len(sys.argv) < 2:
        print('Please enter a filename as a system arg. (Type \'data.csv\')')
        quit()
    else:
        fp = sys.argv[1]
        employees = processFile(pathName(fp))                   #calls function to open and read file, converts returned string into an io buffer
    
    # save the pickle file
    pickle.dump(employees, open('employeesDict.p', 'wb'))  # write binary

    # read the pickle file
    employees_in = pickle.load(open('employeesDict.p', 'rb'))  # read binary
    
    print('\n\nEmployee List:\n')
    for ID in employees_in:
        print('')
        Person.display(employees_in.get(ID))                       #use of display functions

    
    


    