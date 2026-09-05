# === IMPORTATION DES BIBLIOTHÈQUES ===
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import joblib

# === 1. CHARGEMENT DES DONNÉES ===
print("Chargement des données...")
df = pd.read_csv('credit_risk_dataset.csv')

# === 2. PRÉTRAITEMENT (NETTOYAGE) ===
print("Nettoyage des données...")
df['loan_int_rate'] = df['loan_int_rate'].fillna(0)
df = df.dropna()

label_encoders = {}
cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop('loan_status', axis=1)
y = df['loan_status']

# === 3. SÉPARATION TRAIN/TEST ET SCALING ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 4. ENTRAÎNEMENT DU MODÈLE (RANDOM FOREST) ===
print("Entraînement du modèle...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# === 5. ÉVALUATION DU MODÈLE ===
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\nRésultats du Modèle :")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_pred_proba):.4f}")

# === 6. VISUALISATION (ROC CURVE) ===
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Courbe ROC - Credit Risk Model')
plt.legend()
plt.savefig('roc_curve.png')

# === 7. SAUVEGARDE DU MODÈLE, SCALER ET ENCODERS ===
joblib.dump(model, 'credit_scoring_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(list(X.columns), 'feature_columns.pkl')

print("\nModèle, scaler et encoders sauvegardés avec succès!")
print("Fichiers créés: credit_scoring_model.pkl, scaler.pkl, label_encoders.pkl, feature_columns.pkl")
