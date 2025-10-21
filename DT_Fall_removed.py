#96 rows of data after cleaning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline

df = pd.read_csv('Cleaned_Data_Fall_UnknownRemoved.csv')
politics_map = {
    'Strongly liberal': 'Liberal',
    'Mildly liberal': 'Liberal',
    'Strongly conservative': 'Conservative',
    'Mildly conservative': 'Conservative',
    'Neutral': 'Neutral',
    "Don't know/ It's complicated": 'Unaffiliated'
}
df['self politics simplified'] = df['self politics'].map(politics_map)
df['parents politics simplified'] = df['parents politics'].map(politics_map)


print("Unique raw values in self politics:")
print(sorted(df['self politics'].astype(str).unique().tolist()))

def _normalize(s):
    s = str(s)
    s = s.replace("’", "'")        # curly -> straight apostrophe
    s = " ".join(s.split())        # collapse whitespace
    return s.strip().lower()

df['self politics_norm'] = df['self politics'].apply(_normalize)
df['parents politics_norm'] = df['parents politics'].apply(_normalize)

def simplify_politics(s):
    if pd.isna(s):
        return np.nan
    if "liberal" in s:
        return "Liberal"
    if "conservative" in s:
        return "Conservative"
    if "neutral" in s:
        return "Neutral"
    if "don't know" in s or "dont know" in s or "complicated" in s:
        return "Unaffiliated"
    return np.nan

df['self politics simplified'] = df['self politics_norm'].apply(simplify_politics)
df['parents politics simplified'] = df['parents politics_norm'].apply(simplify_politics)


y = df['self politics simplified']
feature_cols = ['gf doctor',
       'not walking daughter down aisle', 'split rent 50 50',
       'paying daughters private school fees', 'lost cat',
       'poorly behaved niece', 'not helping sister on flight',
       'misusing child support', 'making sure fair child support',
       'paying college fees', 'mother-in-law boyfriend',
       'donating to LGBTQ org.', 'cannot drink wife pregnant',
       'bridesmaid sisters hair']
X = df.loc[:,feature_cols]

#preprocess 
numeric_cols = X.select_dtypes(include = [np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
pre = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough',numeric_cols)
    ]
)



#split between training and testing data
labels_order = ["Liberal", 'Conservative', 'Neutral']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=42, stratify=y)


#gini 
clf = DecisionTreeClassifier(criterion="gini",max_depth=4, random_state=42)

pipe = Pipeline([("pre", pre), ("dt", clf)])

#train and eval
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print("Accuracy:", f"{accuracy_score(y_test, y_pred)}")
print(classification_report(y_test,y_pred, labels=labels_order, zero_division=0))

cm = confusion_matrix(y_test, y_pred, labels=labels_order)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_order)
disp.plot(values_format='d')
plt.title('Confusion Matrix self politics (simplified)')
plt.tight_layout
plt.savefig('cmDataFall_removed_AITA.png')

# --- Feature importances (what drove the splits) ---
ohe = pipe.named_steps["pre"].named_transformers_["cat"]
feat_names = list(ohe.get_feature_names_out(categorical_cols)) + numeric_cols
importances = pipe.named_steps["dt"].feature_importances_
imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)
print("\nTop features:\n", imp.head(15))

#plot tree
plt.figure(figsize=(24,12))
plot_tree(
    pipe.named_steps['dt'],
    feature_names = feat_names,
    class_names= pipe.classes_.tolist(),
    filled= True,
    fontsize=9
)

plt.title('Decision Tree gini self politics simplified')
plt.tight_layout()
plt.savefig('DTFall_AITA_removed.png', dpi=200,bbox_inches='tight')


