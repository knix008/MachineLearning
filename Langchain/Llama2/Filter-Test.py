def square(x):
    return x ** x

def double(x):
    return x + x 

def main():
    result = square(2) | double(10)
    print(result)

if __name__ == "__main__": 
    main()