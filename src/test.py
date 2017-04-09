import os,fnmatch



def walk_dirs_exlude(directory, match='*'):
    """Generator function to interate through all subdirectories which do NOT match the given match parameter"""
    for root, dirs, files in os.walk(directory):
        matches = fnmatch.filter(dirs,match)
        for dir in set(dirs).difference(matches):
            yield os.path.join(root,dir)


if __name__ == "__main__":
    for dir in walk_dirs_exlude(".","max"):
        print("{0}".format(dir))