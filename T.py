import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('filename')
    print("End of reading")
    data = df.values
    data = list(map(list, zip(*data)))
    data = pd.DataFrame(data)
    print("Ready to store")
    data.to_csv(r'filename', header=0)