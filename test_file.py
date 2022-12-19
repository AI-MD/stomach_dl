
import os
from pathlib import Path

files=Path("E:/ai_predict/E").resolve().glob('*.*')

images=list(files)

check_class  = { }

for num,img in enumerate(images):
    path = os.path.abspath(img)
    base = os.path.basename(os.path.abspath(img))
    arr = base.split(".")
    print(arr[5])
    print(arr[6])
    check_class[arr[5]] = "check"

print(len(check_class))


