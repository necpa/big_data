import joblib
import numpy as np
from numpy import ndarray
import pandas as pd
from sklearn import pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from statistics import mean

from sklearn.pipeline import FeatureUnion

np.set_printoptions(threshold=10000, suppress=True)
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, make_scorer, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

# Dictionnaire des modèles à comparer
models_to_compare = {
    'CART': DecisionTreeClassifier(random_state=1),
    'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Ad': AdaBoostClassifier(n_estimators=200, random_state=1),
    'RF': RandomForestClassifier(n_estimators=200, random_state=1, n_jobs=-1),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=200, random_state=1),
    'MLP': MLPClassifier(hidden_layer_sizes=(20, 10), random_state=1),
    'Bagging': BaggingClassifier(n_estimators=200, random_state=1, n_jobs=-1),
}


algos = {
    DecisionTreeClassifier: DecisionTreeClassifier(random_state=1),
    KNeighborsClassifier: KNeighborsClassifier(),
    AdaBoostClassifier: AdaBoostClassifier(random_state=1),
    RandomForestClassifier: RandomForestClassifier(random_state=1),
    ExtraTreesClassifier: ExtraTreesClassifier(random_state=1),
    MLPClassifier: MLPClassifier(random_state=1),
    BaggingClassifier: BaggingClassifier(random_state=1),
}


# Fonction de score personnalisée basée sur la précision
# Moyenne entre la précision pour la classe positive et négative
def avg_precision_score_loss(y_true, y_pred):
    return (precision_score(y_true, y_pred) + precision_score(y_true, y_pred, pos_label=0)) / 2

# Moyenne entre la précision et l'accuracy
def avg_precision_accuracy_score(y_true, y_pred):
    return np.mean([precision_score(y_true, y_pred), accuracy_score(y_true, y_pred)])

# Définition des métriques de scoring
scorings = {
    'accuracy': make_scorer(accuracy_score, greater_is_better=True),
    'precision': make_scorer(avg_precision_score_loss, greater_is_better=True),
    'custom': make_scorer(avg_precision_accuracy_score, greater_is_better=True),
    'roc_auc': make_scorer(roc_auc_score, greater_is_better=True)
}

parametres = {
    DecisionTreeClassifier: {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2'],
        'splitter': ['best', 'random'],
    },
    KNeighborsClassifier: {
        'n_neighbors': [3, 5, 10, 20],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2],  # 1: Manhattan, 2: Euclidean
    },
    AdaBoostClassifier: {
        'n_estimators': [50, 100, 200, 500],
        'learning_rate': [0.01, 0.1, 1.0],
        'algorithm': ['SAMME', 'SAMME.R'],
    },
    RandomForestClassifier: {
        'n_estimators': [100, 200, 500, 1000],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [None, 10, 20, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
    },
    ExtraTreesClassifier: {
        'n_estimators': [100, 200, 500],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
    },
    MLPClassifier: {
        'hidden_layer_sizes': [(20,), (40, 20), (50, 25, 10), (100,)],
        'activation': ['relu', 'sigmoid', 'tanh', 'identity'],
        'solver': ['adam', 'sgd', 'lbfgs'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [200, 500, 1000],
    },
    BaggingClassifier: {
        'n_estimators': [50, 100, 200, 500],
        'max_samples': [0.5, 0.7, 1.0],
        'max_features': [0.5, 0.7, 1.0],
        'bootstrap': [True, False],
        'bootstrap_features': [True, False],
    },
}


# Fonction pour normaliser les données
def normalize(x_train, x_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_test = scaler.transform(x_test)
    return X_train, X_test

# Version de normalisation appliquée sur un seul ensemble de données
def normalize_1(x):
    scaler = StandardScaler()
    return scaler.fit_transform(x)

# Fonction pour appliquer une réduction de dimension PCA aux données
def pca(x_train, x_test):
    X_train, X_test = normalize(x_train, x_test)

    pca = PCA(n_components=3)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    X_train = np.hstack((X_train, X_train_pca))
    X_test = np.hstack((X_test, X_test_pca))
    return X_train, X_test

# Version PCA appliquée sur un seul ensemble de données
def pca_1(x):
    X = normalize_1(x)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    return np.hstack((X, X_pca))

# Fonction pour entraîner et évaluer un modèle donné
def train_and_evaluate(model, X_train, y_train, X_test, y_test, strategy):
    # Entrainement du modèle au jeu de données
    model.fit(X_train, y_train)

    # Prediction du jeu de donnée d'entrainement
    y_train_predict = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_predict)

    # Prediction du jeu de donnée de test
    y_test_predict = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_predict)

    precision_result = precision_score(y_test, y_test_predict)
    mean_score = mean([test_accuracy, precision_result])

    print(f'--------------------------------{model.__class__.__name__}({strategy})--------------------------------')
    print(f'Train accuracy: {train_accuracy:.2f}')
    print(f'Test accuracy: {test_accuracy:.2f}')
    print(f'Precision score: {precision_result:.2f}')
    print(f'Mean score: {mean_score:.2f}')
    print('')

    return mean_score


# Sélection du meilleur modèle selon différentes stratégies (normal, normalisé, PCA)
def get_best_models(models, x_train, x_test, y_train, y_test):
    strategies = {'normal', 'normalisé', 'pca'}
    best_score = 0
    best_strategy = None
    best_model = None
    best_data = (x_train, x_test)
    for model in models:
        for strategy in strategies:
            X_train, X_test = x_train, x_test

            match strategy:
                case 'pca':
                    X_train, X_test = pca(x_train, x_test)
                case 'normalisé':
                    X_train, X_test = normalize(x_train, x_test)

            score = train_and_evaluate(model, X_train, y_train, X_test, y_test, strategy)
            if score > best_score:
                best_score = score
                best_strategy = strategy
                best_model = model
                best_data = (X_train, X_test)

    print(f'--------------------------------Best Combinaison--------------------------------')
    print(f'Best score: {best_score:.2f}')
    print(f'Best model: {best_model.__class__.__name__}')
    print(f'Best strategy: {best_strategy}')
    return best_model, best_strategy, best_data


def get_best_model_cross_validation(models, x, y, scoring):
    strategies = {'normal', 'normalisé', 'pca'}
    best_score = -np.inf
    best_strategy = None
    best_model = None
    best_data = x
    for model in models.values():
        for strategy in strategies:
            X = x
            match strategy:
                case 'pca':
                    X = pca_1(X)
                case 'normalisé':
                    X = normalize_1(X)

            score = cross_val_score(estimator=model, X=X, y=y, cv=10, scoring=scoring)
            score_mean = np.mean(score)
            print(
                f'--------------------------------{model.__class__.__name__}({strategy})--------------------------------')
            print(f'Mean score: {score_mean:.2f}')
            print('')
            if score_mean > best_score:
                best_score = score_mean
                best_strategy = strategy
                best_model = model
                best_data = X

    print(f'--------------------------------Best Combinaison--------------------------------')
    print(f'Best score: {best_score:.2f}')
    print(f'Best model: {best_model.__class__.__name__}')
    print(f'Best strategy: {best_strategy}')
    return best_model, best_strategy, best_data

# Cette fonction permet de trier les variables en fonction de leur importance
def sort_variables(X_train, y_train, verbose):
    clf = RandomForestClassifier(n_estimators=1000, random_state=1)
    clf.fit(X_train, y_train)
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    if verbose:
        std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
        features = X_train.shape[1]
        padding = np.arange(X_train.size / len(X_train)) + 0.5
        plt.barh(padding, importances[sorted_idx], xerr=std[sorted_idx], align='center')
        plt.yticks(padding, features[sorted_idx])
        plt.xlabel("Relative Importance")
        plt.title("Variable Importance")
        plt.show()
    return sorted_idx


def select_variables(model, X_train, X_test, y_train, y_test, verbose):
    sorted_idx = sort_variables(X_train, y_train, verbose)
    scores = np.zeros(X_train.shape[1] + 1)
    for f in np.arange(0, X_train.shape[1] + 1):
        X1_f = X_train[:, sorted_idx[:f + 1]]
        X2_f = X_test[:, sorted_idx[:f + 1]]
        model.fit(X1_f, y_train)
        y_test_pred = model.predict(X2_f)
        scores[f] = np.round(accuracy_score(y_test, y_test_pred), 3)

    if verbose:
        plt.plot(scores)
        plt.xlabel("Nombre de Variables")
        plt.ylabel("Accuracy")
        plt.title("Evolution de l'accuracy en fonction des variables")
        plt.show()
    return np.argmax(scores), sorted_idx


def selection_variables(model, X, Y, verbose):
    sorted_idx = sort_variables(X, Y, verbose=verbose)
    scores = np.zeros(X.shape[1] + 1)
    for f in np.arange(0, X.shape[1] + 1):
        X_f = X[:, sorted_idx[:f + 1]]
        model.fit(X_f, Y)
        y_pred = model.predict(X_f)
        scores[f] = np.round(accuracy_score(Y, y_pred), 3)

    if verbose:
        plt.plot(scores)
        plt.xlabel("Nombre de Variables")
        plt.ylabel("Accuracy")
        plt.title("Evolution de l'accuracy en fonction des variables")
        plt.show()
    return np.argmax(scores)

# Fonction pour effectuer un grid search et trouver les meilleurs paramètres
def best_parameters(model, param_grid, X, y, scoring):
    grid_search = GridSearchCV(model, param_grid, cv=10, scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

# Fonction pour créer un pipelin pour généraliser la stategy et le modèle
def creation_pipeline(model, x, y, strategy, max_features):
    p = None
    clf = RandomForestClassifier(n_estimators=1000, random_state=1)
    match strategy:
        case 'normal':
            p = pipeline.Pipeline([
                ('FS', SelectFromModel(clf, max_features=max_features)),
                ('Classifier', model)
            ])
        case 'normalisé':
            p = pipeline.Pipeline([
                ('SS', StandardScaler()),
                ('FS', SelectFromModel(clf, max_features=max_features)),
                ('classifier', model)
            ])
        case 'pca':
            p = pipeline.Pipeline([
                ('SS', StandardScaler()),
                ('PCA', FeatureUnion([('pca', PCA(n_components=3)), ('SS', StandardScaler())])),
                ('FS', SelectFromModel(clf, max_features=max_features)),
                ('Classifier', model)
            ])
        case _:
            AssertionError('Strategy not recognized')

    p.fit(x, y)
    joblib.dump(p, open('pipeline.joblib', "wb"))

# Fonction d'entrainement pour trouver le meilleur modèle, les meilleurs hyperparamètres et les meilleurs stratégies

def train(X: ndarray, Y: ndarray, scoring: str):
    s = scorings[scoring]
    if Y.ndim == 1:
        models = models_to_compare
        best_model, best_strategy, best_data = get_best_model_cross_validation(models, X, Y, s)

        parametres_alg = parametres[type(best_model)]
        best_alg = algos[type(best_model)]
        best_model = best_parameters(best_alg, parametres_alg, best_data, Y, s)

        nb_selected_features = selection_variables(best_model, best_data, Y, True)

        creation_pipeline(best_model, X, Y, best_strategy, nb_selected_features)

        return best_model, best_strategy, nb_selected_features


### Apprentissage supervisé : Données heterogènes


def delete_missing_values(X, y):
	valid_rows = ~np.isnan(X).any(axis=1)
	X_clean = X[valid_rows]
	y_clean = y[valid_rows]
	return X_clean, y_clean


def load_and_prepare_data(filepath, verbose=False):
	data = pd.read_csv(filepath, header=None, delimiter="\t")
	data_array = data.values
	X = data_array[:, :-1]
	y = data_array[:, -1]

	numeric_cols = []
	categorical_cols = []
	for col in range(X.shape[1]):
		try:
			X[:, col] = np.where(X[:, col] == '?', np.nan, X[:, col])
			X[:, col].astype(float)
			numeric_cols.append(col)
		except ValueError:
			categorical_cols.append(col)
			continue

	print(f'Colonnes numériques : {numeric_cols}')
	print(f'Colonnes catégorielles : {categorical_cols}')

	X_num = X[:, numeric_cols]
	X_num = X_num.astype(float)

	y = np.where(y == '+', 1, 0)

	if verbose :
		print(f'Taille de l\'échantillon:, {X_num.shape}') 
		print("Nombre d'exemples positifs et négatifs:")
		plt.hist(y)
		plt.show()
	print(f'Nombre d\'exemples positifs : {np.sum(y == 1)}')
	print(f'Nombre d\'exemples négatifs : {np.sum(y == 0)}')

	return X, y, numeric_cols, categorical_cols

# Fonction pour gérer les valeurs manquantes

def handle_missing_values(X, y, numeric_cols, categorical_cols):
    X_num = X[:, numeric_cols].astype(float)

    #Imputer les valeurs manquantes dans les colonnes numériques avec la moyenne
    imp_num = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_num = imp_num.fit_transform(X_num)

    X_cat = X[:, categorical_cols].astype(str)

	#Convertir les valeurs catégorielles en entiers
    for col_id in range(X_cat.shape[1]):
        unique_val, val_idx = np.unique(X_cat[:, col_id], return_inverse=True)
        X_cat[:, col_id] = val_idx

    X_cat = X_cat.astype(float)

    #Imputer les valeurs manquantes dans les colonnes catégorielles avec la valeur la plus fréquente
    imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    X_cat = imp_cat.fit_transform(X_cat)

    #supprimer les lignes avec des valeurs manquantes
    valid_rows = ~np.isnan(X_num).any(axis=1) & ~np.isnan(X_cat).any(axis=1)
    X_num = X_num[valid_rows]
    X_cat = X_cat[valid_rows]
    y = y[valid_rows]
    
    return X_num, X_cat, y

# Fonction pour encoder les valeurs catégorielles et normaliser les valeurs numériques

def encode_and_normalize(X_num, X_cat):
    encoder = OneHotEncoder()
    X_cat_bin = encoder.fit_transform(X_cat).toarray()

    X_num_scaled = normalize_1(X_num)

    X = np.hstack((X_num_scaled, X_cat_bin))
    return X
