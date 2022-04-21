import pandas as pd

def txt2list(input_file):
    return open(input_file,'r').read().splitlines()


def list2txt(lst, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for line in lst:
            line = line.rstrip('\n')
            f.write(f"{line}\n")


def csv_to_excel(source_file, des_file):
    read_file = pd.read_csv(source_file)
    read_file.to_excel(des_file, index=None, header=True)
