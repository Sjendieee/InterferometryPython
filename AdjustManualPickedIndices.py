from SwellingFromAbsoluteIntensity import saveDataToFile
import glob
import os
import numpy as np

def main():
    """
           Read numbers from a .txt file and convert them to a single array, with a certain offset.
           The input data can be written as individual numbers on separate lines
           or on one line divided by a space.

           Parameters:
           - file_path (str): The path to the .txt file.

           Returns:
           - list of int: The array containing the read numbers.
           """
    offset = -200        #nr with which all the numbers will be increased
    source = "D:\\2023_11_13_PLMA_Dodecane_Basler5x_Xp_1_24S11los_misschien_WEDGE_v2\\Swellingimages"
    csvList = [f for f in glob.glob(os.path.join(source, f"*.txt"))]
    for idx, file_path in enumerate(csvList):
        numbers = []
        root_path = os.path.split(file_path)
        with open(file_path, 'r') as file:
            # Read lines from the file
            lines = file.readlines()

            # Check if the data is on separate lines or on one line
            if len(lines) == 1 and ' ' in lines[0]:
                # Data is on one line, split by space
                numbers = [int(num) for num in lines[0].split()]
            else:
                # Data is on separate lines
                numbers = [int(num) for num in lines]
        numbers = list(np.array(numbers) + offset)

        saveDataToFile(numbers, root_path[0], root_path[1])
    return

if __name__ == '__main__':
    main()
    exit()