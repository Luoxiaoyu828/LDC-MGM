import os


def create_folder(path):
    """
    创建文件夹
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
        print(path + 'created successfully!')


if __name__ == '__main__':
    pass