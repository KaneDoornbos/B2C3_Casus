import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Laden van de dataset
dataset = pd.read_csv('./CasusData.csv')  # Vervang 'jouw_bestandsnaam.csv' door de werkelijke bestandsnaam

# 1. Negeren van WAP's met een waarde van 100
# Maak een masker om rijen met WAP-waarden van 100 te negeren
exclude_columns = ['WAP494', 'WAP495', 'WAP496', 'WAP497', 'WAP498', 'WAP499', 'WAP500', 'WAP501', 'WAP502', 'WAP503', 'WAP504', 'WAP505', 'WAP506', 'WAP507', 'WAP508', 'WAP509', 'WAP510', 'WAP511', 'WAP512', 'WAP513', 'WAP514', 'WAP515', 'WAP516', 'WAP517', 'WAP518', 'WAP519', 'WAP520']
mask = ~(dataset[exclude_columns] == 100).any(axis=1)

# Controleer of er voldoende gegevens overblijven na filtering
if mask.sum() == 0:
    print("Er zijn geen geldige samples over na filtering.")
else:
    # Pas het masker toe op de dataset
    dataset_filtered = dataset[mask]

    # Behandeling van ontbrekende waarden
    imputer = SimpleImputer(strategy='mean')
    dataset_imputed = pd.DataFrame(imputer.fit_transform(dataset_filtered), columns=dataset.columns)

    # 3. Feature scaling (normalisatie)
    # Hier gebruiken we StandardScaler om de features te normaliseren.
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(dataset_imputed.iloc[:, :520])

    # 4. K-means clustering voor groepering van locaties
    kmeans = KMeans(n_clusters=3, random_state=42)
    dataset['LOCATION_CLUSTER'] = kmeans.fit_predict(features_scaled)

    # 5. Nieuwe kolom: WALKING_PATTERN (bijvoorbeeld op basis van locatieverandering)
    dataset['WALKING_PATTERN'] = (dataset['LATITUDE'].diff() != 0) | (dataset['LONGITUDE'].diff() != 0)
    dataset['WALKING_PATTERN'] = dataset['WALKING_PATTERN'].astype(int)

    # 6. Groeperen op individuele gebruikers
    grouped_data = dataset.groupby('USERID').agg({
        'WALKING_PATTERN': 'max',  # Aggregeer het wandelpatroon (max waarde over tijd)
        'LOCATION_CLUSTER': 'max',  # Aggregeer de locatiecluster (max waarde over tijd)
        # Voeg andere gewenste aggregaties toe voor extra informatie
    }).reset_index()

    # Doelvariabele
    target = grouped_data['WALKING_PATTERN']

    # Features voor voorspelling (bijvoorbeeld locatiecluster)
    features_grouped = grouped_data[['LOCATION_CLUSTER']]

    # Split de data in trainings- en testsets
    X_train, X_test, y_train, y_test = train_test_split(features_grouped, target, test_size=0.2, random_state=42)

    # Train een Random Forest Classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    # Voorspel de WALKING_PATTERN op de testset
    predictions = classifier.predict(X_test)

    # Evaluatie van het model
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy:.2f}')

    # Gedetailleerde evaluatie
    print('Classification Report:\n', classification_report(y_test, predictions))

# Print de unieke waarden in elke WAP-kolom
for wap_col in dataset.columns:
    if wap_col.startswith('WAP'):
        unique_values = dataset[wap_col].unique()
        print(f'{wap_col}: {unique_values}')