import os
import sys
import argparse

def remove_bmp_files(directory):
    """
    Recursively removes all .bmp files from the specified directory.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.bmp'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove all BMP files from a directory recursively')
    parser.add_argument('directory', help='Path to the directory to process')
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        sys.exit(1)
        
    remove_bmp_files(args.directory)
    print("BMP file removal complete.")
