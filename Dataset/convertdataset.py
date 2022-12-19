import os
from shutil import copyfile

def search(dirname,dst):
    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                search(full_filename,dst)
            else:
                ext = os.path.splitext(full_filename)[-1]
                if ext == '.jpg':
                    label=full_filename[-6:-4]
                    dst_path=os.path.join(dst,label)
                    dst_path=os.path.join(dst_path,os.path.basename(full_filename))
                    #print(dst_path)
                    copyfile(full_filename, dst_path)
                    #print(full_filename)
    except PermissionError:
        pass

search("E:/stomach/1028_dataset", "E:/stomach/project/stomach_project_ver2/stomach_mid_new/train")