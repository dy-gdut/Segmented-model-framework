import shutil
from glob import glob


# 避免程序的多次运行生成垃圾文件，下面函数可以设置每个文件夹最多保留keep_num个文件夹
def del_directory(root, keep_num=5):
    directory_paths = glob(root+"/*")
    length = len(directory_paths)
    if length == 0 | length <= keep_num:
        return
    # 按时间排序，往后切片
    del_directory_paths = sorted(directory_paths)[:length - keep_num % length]
    for del_directory_path in del_directory_paths:
        shutil.rmtree(del_directory_path)
