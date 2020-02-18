import os
import glob


def add_status(unsolved__list: list, solved_list: list, file_path: str):
    files = glob.glob(file_path + '*')
    for i in unsolved__list:
        for f in files:
            if f == file_path + f'Q{i}.py':
                path = file_path + f'Q{i}.py'
                os.rename(path, os.path.join(file_path, f'Q{i}_not_completed.py'))
    for i in solved_list:
        for f in files:
            if f == file_path + f'Q{i}_not_completed.py':
                path = file_path + f'Q{i}_not_completed.py'
                os.rename(path, os.path.join(file_path, f'Q{i}.py'))


if __name__ == '__main__':
    unsolved_num = [5, 8, 9, 10]
    solved_num = [1, 2, 3, 4]
    add_status(unsolved_num, solved_num, './Q1-10/')

    unsolved_num = [19]
    solved_num = []
    add_status(unsolved_num, solved_num, './Q11-20/')
