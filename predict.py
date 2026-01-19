# Modified predict.py
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

def predict_autoencoder(user_input, model, encoders, scaler, threshold):
    input_df = pd.DataFrame([user_input])

    for col in encoders:
        if user_input[col] not in encoders[col].classes_:
            if 'unknown' in encoders[col].classes_:
                input_df[col] = encoders[col].transform(['unknown'])
            else:
                input_df[col] = encoders[col].transform([encoders[col].classes_[0]])
        else:
            input_df[col] = encoders[col].transform([user_input[col]])

    input_df[['Amount']] = scaler.transform(input_df[['Amount']])
    input_tensor = torch.tensor(input_df.values, dtype=torch.float32)

    model.eval()
    output = model(input_tensor).detach().numpy()
    error_vector = np.abs(input_df.values - output)
    total_error = np.mean(np.power(input_df.values - output, 2), axis=1)

    prediction = "FRAUDULENT" if total_error > threshold else "LEGITIMATE"
    top_indices = error_vector[0].argsort()[-3:][::-1]
    top_features = [(input_df.columns[i], error_vector[0][i]) for i in top_indices]

    return prediction, total_error[0], top_features

def predict_gnn(new_txn_raw, model, encoders, scaler):
    try:
        import copy
        df_graph = pd.read_pickle("df_graph.pkl")
    except:
        df_graph = pd.DataFrame({
            "Card_Number": [1, 2, 3],
            "Merchant": [0, 1, 0],
            "IP_Address": [1, 1, 2],
            "Amount": [0.1, 0.2, 0.3],
            "Cardholder_Name": [0, 1, 2],
            "Transaction_Type": [0, 0, 1],
            "Device_Used": [0, 1, 0],
            "Location": [0, 1, 2],
            "Fraud_Label": [0, 0, 1],
            "Transaction_ID": [1, 2, 3],
            "Timestamp": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
        })

    df_temp = copy.deepcopy(df_graph)
    new_txn = {
        "Amount": scaler.transform([[new_txn_raw["Amount"]]])[0][0],
        "Merchant": encoders["Merchant"].transform([new_txn_raw["Merchant"]])[0] if new_txn_raw["Merchant"] in encoders["Merchant"].classes_ else 0,
        "Cardholder_Name": encoders["Cardholder_Name"].transform([new_txn_raw["Cardholder_Name"]])[0] if new_txn_raw["Cardholder_Name"] in encoders["Cardholder_Name"].classes_ else 0,
        "Transaction_Type": encoders["Transaction_Type"].transform([new_txn_raw["Transaction_Type"]])[0] if new_txn_raw["Transaction_Type"] in encoders["Transaction_Type"].classes_ else 0,
        "Device_Used": encoders["Device_Used"].transform([new_txn_raw["Device_Used"]])[0] if new_txn_raw["Device_Used"] in encoders["Device_Used"].classes_ else 0,
        "Location": encoders["Location"].transform([new_txn_raw["Location"]])[0] if new_txn_raw["Location"] in encoders["Location"].classes_ else 0,
        "Card_Number": df_graph["Card_Number"].max() + 1,
        "IP_Address": df_graph["IP_Address"].max() + 1,
        "Fraud_Label": 0,
        "Transaction_ID": 999999,
        "Timestamp": pd.to_datetime("2025-01-01 00:00:00")
    }

    df_temp = pd.concat([df_temp, pd.DataFrame([new_txn])], ignore_index=True)

    for col in ['Card_Number', 'Merchant', 'IP_Address']:
        df_temp[col] = LabelEncoder().fit_transform(df_temp[col])

    fraud_neighbors = []
    new_idx = df_temp.index[-1]
    for attr in ['Card_Number', 'Merchant', 'IP_Address']:
        matching = df_temp[(df_temp[attr] == df_temp.loc[new_idx, attr]) & (df_temp.index != new_idx)]
        frauds = matching[matching['Fraud_Label'] == 1]
        if not frauds.empty:
            fraud_neighbors.append(attr)

    edges = set()
    for attr in ['Card_Number', 'Merchant', 'IP_Address']:
        for val in df_temp[attr].unique():
            idxs = df_temp[df_temp[attr] == val].index.tolist()
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    edges.add((idxs[i], idxs[j]))
                    edges.add((idxs[j], idxs[i]))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    X_tensor = torch.tensor(df_temp.drop(['Transaction_ID', 'Fraud_Label', 'Timestamp'], axis=1).values, dtype=torch.float32)
    y_tensor = torch.tensor(df_temp['Fraud_Label'].values, dtype=torch.long)

    data_input = Data(x=X_tensor, edge_index=edge_index, y=y_tensor)
    model.eval()
    pred = model(data_input).argmax(dim=1)
    prediction = pred[-1].item()
    result = "FRAUDULENT" if prediction == 1 else "LEGITIMATE"

    return result, fraud_neighbors