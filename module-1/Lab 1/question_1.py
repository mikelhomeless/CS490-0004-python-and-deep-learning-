
a=input('Please type the string: ')
j=0

while((j)<len(a)):
    i=0
    while(i<=j):
        if(a[i-1]==a[j] and i!=0):
            a=a[:j]+a[j+1:]

            j=j-1
            break;
        i=i+1
    j=j+1
print(a)