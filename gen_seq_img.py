import os
import shutil

def reorganize_images_by_view_per_folder(root_dir, output_base_dir="reorganized_by_view_per_folder"):
    """
    读取指定根目录下的所有图片，并根据原始文件夹和视角进行重新组织。
    为每个原始小文件夹创建8个新的视角子文件夹。

    Args:
        root_dir (str): 包含 'origin_multi_view_step' 文件夹的根目录。
                        例如，如果 'origin_multi_view_step' 在当前目录，则 root_dir='.'
        output_base_dir (str): 用于存放重新组织图片的输出目录名称。
                                默认为 'reorganized_by_view_per_folder'。
    """

    source_dir = os.path.join(root_dir, "origin_multi_view_step")

    if not os.path.exists(source_dir):
        print(f"错误: 源目录 '{source_dir}' 不存在。请检查路径。")
        return

    # 创建输出根目录
    output_full_base_dir = os.path.join(root_dir, output_base_dir)
    os.makedirs(output_full_base_dir, exist_ok=True)
    print(f"创建输出根目录: {output_full_base_dir}")

    num_views = 8  # 0-7，共8个视角

    # 遍历所有小文件夹 (e.g., 001, 002, ..., 400)
    for folder_name in sorted(os.listdir(source_dir)):
        original_folder_path = os.path.join(source_dir, folder_name)

        # 确保它是一个目录
        if os.path.isdir(original_folder_path):
            print(f"处理原始文件夹: {folder_name}")

            # 为当前原始文件夹创建8个视角子文件夹
            current_folder_view_dirs = {}
            for i in range(num_views):
                # 新的子文件夹命名格式：例如 "009_view_0"
                view_folder_name = f"{folder_name}_view_{i}"
                view_path = os.path.join(output_full_base_dir, view_folder_name)
                os.makedirs(view_path, exist_ok=True)
                current_folder_view_dirs[i] = view_path
                # print(f"  创建子视角文件夹: {view_path}") # 可以取消注释查看创建过程

            # 遍历当前原始文件夹内的图片
            for image_name in sorted(os.listdir(original_folder_path)):
                # 检查文件是否是 .png 图片
                if image_name.endswith(".png"):
                    # 示例文件名: 009_step0_0.png
                    parts = image_name.split('_')
                    if len(parts) >= 3:
                        try:
                            # 获取视角编号 (例如，从 '0.png' 中提取 0)
                            view_str = parts[-1].replace('.png', '')
                            view_angle = int(view_str)

                            if 0 <= view_angle < num_views:
                                source_image_path = os.path.join(original_folder_path, image_name)
                                # 在新的组织形式下，文件名不需要再添加原始文件夹前缀，因为每个文件夹已经对应一个原始文件夹
                                destination_path = os.path.join(current_folder_view_dirs[view_angle], image_name)

                                shutil.copy2(source_image_path, destination_path)
                                # print(f"    复制: {image_name} -> {destination_path}")
                            else:
                                print(f"  警告: 文件 '{image_name}' 包含无效视角编号: {view_angle}。跳过。")
                        except ValueError:
                            print(f"  警告: 无法从文件 '{image_name}' 中解析视角编号。跳过。")
                    else:
                        print(f"  警告: 文件 '{image_name}' 的命名格式不符合预期。跳过。")
                else:
                    pass # 跳过非png文件

    print("\n图片重新组织完成！")
    print(f"所有图片现在存储在 '{output_full_base_dir}' 目录下，并按原始文件夹和视角分类。")

# --- 如何使用 ---
# 将 'root_directory' 设置为包含 'origin_multi_view_step' 文件夹的父目录。
# 例如，如果 'origin_multi_view_step' 在 '/home/baiqiao/spatial_generation/Spatial_Mental_Mani/data' 目录下
root_directory = '/home/baiqiao/spatial_generation/Spatial_Mental_Mani/data'
reorganize_images_by_view_per_folder(root_directory)

# 你也可以指定不同的输出目录名称，例如 'my_new_organized_images'
# reorganize_images_by_view_per_folder(root_directory, output_base_dir="my_new_organized_images")