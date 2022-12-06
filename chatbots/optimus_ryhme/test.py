#Python program to explain os.mkdir() method
  
# importing os module
import os
import pathlib
  
# Directory
directory = f".\models\MovieDataSet\somethinafg"

# print("hi")
# print(os.getcwd())
# print(os.getcwdb())
# print(os.get_exec_path())


if("optimus_ryhme" in os.getcwd()):
    directory = os.path.abspath("../../../models")
else:
    directory = os.path.abspath("./models")



print(directory)


if os.path.exists(directory):
    print(f"Directory {directory} already exitsts")
else:
    os.mkdir(directory)
    print("created directory")