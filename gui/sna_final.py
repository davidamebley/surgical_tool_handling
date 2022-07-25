import pandas as pd
# "Source", "Target", "Type", "Id", "Label", "Weight", "weight usd" 1057178,33
data_frame = pd.read_csv("../../../../../Period 4/2022/SNA/Group Proj/Group Gephi/data.csv")

filtered_df = data_frame[data_frame['weight usd'].notnull()]

new_data = filtered_df.loc[(filtered_df["weight usd"] >= 5000000), ["Source", "Target", "Type", "Id", "Label", "Weight", "weight usd"]]

print(new_data.shape)

new_data.to_csv("../../../../../Period 4/2022/SNA/Group Proj/Group Gephi/new_data.csv", encoding='utf-8', index=False)

