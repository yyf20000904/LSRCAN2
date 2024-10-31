import os
import shutil

def copy_and_rename_images(src_folder, dest_folder):
    # 获取源文件夹下的所有文件夹
    subdirectories = [d for d in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, d))]
    print(subdirectories)

    # 遍历每个子文件夹
    for subdir in subdirectories:
        # 构建源文件夹中每个子文件夹的完整路径
        src_subfolder = os.path.join(src_folder, subdir)

        # 获取子文件夹中的所有图片文件
        image_files = [f for f in os.listdir(src_subfolder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif','.bmp'))]

        # 遍历每个图片文件
        for i, image_file in enumerate(image_files):
            # 构建源图片文件路径
            src_image_path = os.path.join(src_subfolder, image_file)

            # 构建目标图片文件路径
            dest_image_name = f"{subdir}_image_{i+1}{os.path.splitext(image_file)[1]}"
            dest_image_path = os.path.join(dest_folder, dest_image_name)

            # 复制并重命名图片文件
            shutil.copy(src_image_path, dest_image_path)
   

if __name__ == "__main__":
    # 源文件夹和目标文件夹的路径
    source_folder = "img"
    destination_folder = "realtrain"

 
    # 复制并重命名图片
    copy_and_rename_images(source_folder, destination_folder)
