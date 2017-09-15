import sys
input_arg=sys.argv
arg1=input_arg[1]

f = open(arg1,"r")
a = f.read().split()
b = dict()
c = list()

for i in a:
    b[i]=1
    if i not in c:
        c.append(i)

for key in b:
    b[key] = a.count(key)
    
fout = open("Q1.txt","w")
fout.write('')

for i in c:
    if c.index(i) != len(c)-1:
        fout.write(str(i)+' '+str(c.index(i))+' '+str(b[i])+'\n')
    else:
        fout.write(str(i)+' '+str(c.index(i))+' '+str(b[i]))
    
f.close()
fout.close()