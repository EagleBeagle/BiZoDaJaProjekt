import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


sentiment140 = pd.read_csv('../training.1600000.processed.noemoticon.csv', encoding="ISO-8859-1")

sentiment140.columns = ['label', 'id', 'date', 'flag', 'user', 'text']
sentiment140['length'] = sentiment140['text'].apply(len)
sns.barplot('label','length',data = sentiment140,palette='PRGn')
plt.title('Average Word Length vs Label')
plt.show()

fig2 = sns.countplot(x= 'label',data = sentiment140)
plt.title('Label Counts')
plot = fig2.get_figure()
plt.show()
