import os

ai_list =[]
valid_list =[]


# for f_name in os.listdir("E:/ai_predict/D"):
#     if f_name.endswith(".bmp"):
#         ai_list.append(f_name)
#
#
#
#
# for f_name in os.listdir("E:/ai_predict/E"):
#     if f_name.endswith(".bmp"):
#         ai_list.append(f_name)

for f_name in os.listdir("E:/ai_predict/S1"):
    if f_name.endswith(".bmp"):
        ai_list.append(f_name)
#
# for f_name in os.listdir("E:/ai_predict/S2"):
#     if f_name.endswith(".bmp"):
#         ai_list.append(f_name)
#
# for f_name in os.listdir("E:/ai_predict/S3"):
#     if f_name.endswith(".bmp"):
#         ai_list.append(f_name)
#
# for f_name in os.listdir("E:/ai_predict/S4"):
#     if f_name.endswith(".bmp"):
#         ai_list.append(f_name)
#
# for f_name in os.listdir("E:/ai_predict/S5"):
#     if f_name.endswith(".bmp"):
#         ai_list.append(f_name)
#
# for f_name in os.listdir("E:/ai_predict/S6"):
#     if f_name.endswith(".bmp"):
#         ai_list.append(f_name)
#
# for f_name in os.listdir("E:/ai_predict/X"):
#     if f_name.endswith(".bmp"):
#         ai_list.append(f_name)






for f_name in os.listdir("E:/valid_data/D"):
    if f_name.endswith(".bmp"):
        valid_list.append(f_name)

#
# #
for f_name in os.listdir("E:/valid_data/E"):
    if f_name.endswith(".bmp"):
        valid_list.append(f_name)

# for f_name in os.listdir("E:/valid_data/S1"):
#     if f_name.endswith(".bmp"):
#         valid_list.append(f_name)


for f_name in os.listdir("E:/valid_data/S2"):
    if f_name.endswith(".bmp"):

        valid_list.append(f_name)
for f_name in os.listdir("E:/valid_data/S3"):
    if f_name.endswith(".bmp"):
        valid_list.append(f_name)

for f_name in os.listdir("E:/valid_data/S4"):
    if f_name.endswith(".bmp"):
        valid_list.append(f_name)

for f_name in os.listdir("E:/valid_data/S5"):
    if f_name.endswith(".bmp"):
        valid_list.append(f_name)


for f_name in os.listdir("E:/valid_data/S6"):
    if f_name.endswith(".bmp"):
        valid_list.append(f_name)

for f_name in os.listdir("E:/valid_data/X"):
    if f_name.endswith(".bmp"):
        valid_list.append(f_name)


print(len(ai_list),len(valid_list))
count = 0
for valid_value in valid_list:
    for ai_value in ai_list:
        if ai_value == valid_value:
            count = count + 1
            #if ai_value in ai_list:
            # print(ai_value, valid_value)
            #ai_list.remove(ai_value)

print(count)


#
# import os
# import shutil
#
# results = []
# for (path, dir, files) in os.walk("E:/ai_predict"):
#     for filename in files:
#         ext = os.path.splitext(filename)[-1]
#         if ext == '.bmp':
#             if filename in ai_list:
#                 results.append(os.path.join(path,filename))
#                 basename_2 = os.path.basename(path)
#                 if os.path.isdir(os.path.join("E:/ai_predict_new", basename_2)):
#                     pass
#                 else:
#                     os.mkdir(os.path.join("E:/ai_predict_new", basename_2))
#
#
#
# print(len(results))
#
#
# print(results)
#
# for filePath in results:
#     basename = os.path.basename(filePath)
#     dir = os.path.dirname(filePath)
#     basename_2 = os.path.basename(dir)
#     shutil.copy2(filePath, os.path.join("E:/ai_predict_new", basename_2))

    #




#
#
# print(count)
# check_class = {}
# flag ={
#        "E" : False,
#        "S1" : False,
#        "S2": False,
#        "S3": False,
#        "S4" : False,
#        "S5" : False,
#        "D": False,
#        }


#
# for f_name in os.listdir("E:/ai_predict/X"):
#     base = os.path.basename(f_name)
#     arr = base.split(".")
#     check_class[arr[5]] = flag
# class_name = ["E","S1","S2","S3","S4","S5","D"]
#
#
# for name in class_name:
#     for f_name in os.listdir("E:/ai_predict/"+name):
#         base = os.path.basename(f_name)
#         arr = base.split(".")
#
#         if check_class.get(arr[5]):
#             check_dic = check_class[arr[5]]
#             check_dic[name] = True
#         check_class[name] = check_dic
#
#
# print(len(check_class))



