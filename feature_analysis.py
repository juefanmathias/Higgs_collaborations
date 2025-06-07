from HiggsML.datasets import download_dataset
from utils import histogram_dataset
import matplotlib.pyplot as plt

data = download_dataset(
    "blackSwan_data"
)  # change to "blackSwan_data" for the actual data
# load train set
data.load_train_set()
data_set = data.get_train_set()
labels = data_set["labels"]
weights = data_set["weights"]

feature_columns = [col for col in data_set.columns if col.startswith("PRI_") or col.startswith("DER_")]

for i in range(0, len(feature_columns), 4):
    subset = feature_columns[i:i+4]
    histogram_dataset(data_set, labels, weights, columns=subset)
    plt.show()  # 非 notebook 环境下必须加上这一句
