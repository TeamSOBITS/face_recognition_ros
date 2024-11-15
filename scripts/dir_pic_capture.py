import os


path_ = "/home/sobits/catkin_ws/src/test"

print(os.path.isdir(path_))
print(os.path.isfile(path_))
print(os.listdir(path_))
# for s in os.listdir(path_):
#     print(s, s.split(".")[0])
