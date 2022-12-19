import shutil
import os

f=open("../error_stomach.txt",'r')

while True:
    line=f.readline()
    filename = os.path.basename(line)
    filename=filename.split("\n")[0]
    print(filename)
    src="../data_new/"
    dst="../data_new/bad/"
    shutil.move(src+filename,dst+filename)
    if not line: break

f.close()