import pandas as pd
import torch

if __name__ == '__main__':
    df = pd.read_csv('anon_4_4_hangzhou_real_6539.csv')
    print(df)

    # Convert DataFrame to numpy array
    adj_matrix = df.values
    print(adj_matrix)
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.int8)
    print(adj_tensor.shape)


