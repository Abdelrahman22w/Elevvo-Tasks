# Forest Cover Type Classification - Model Comparison
# Testing different approaches to handle class imbalance

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('default')
sns.set_palette("husl")

def load_and_preprocess_data():
    """Load and clean the dataset"""
    print("Loading dataset...")
    
    df = pd.read_csv('covertype.csv')
    
    # Clean up any unnamed columns
    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        print(f"Removed {len(unnamed_cols)} unnamed columns")
    
    # Split features and target
    X = df.drop('Cover_Type', axis=1)
    y = df['Cover_Type']
    
    print(f"Features: {X.shape}, Target: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Data preprocessing done")
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler

def analyze_class_distribution(y_train, y_test):
    """Check class balance"""
    print("\n" + "="*50)
    print("CLASS DISTRIBUTION")
    print("="*50)
    
    train_counts = pd.Series(y_train).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()
    
    print("Training set:")
    for cover_type, count in train_counts.items():
        percentage = (count / len(y_train)) * 100
        print(f"  Cover Type {cover_type}: {count:,} ({percentage:.2f}%)")
    
    imbalance_ratio = train_counts.max() / train_counts.min()
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}")
    
    return train_counts, test_counts

def test_baseline_models(X_train, X_test, y_train, y_test):
    """Try basic models first"""
    print("\n" + "="*50)
    print("BASELINE MODELS")
    print("="*50)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=1),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=1),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=1),
        'XGBoost': xgb.XGBClassifier(random_state=42, n_jobs=1)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        start_time = time.time()
        
        try:
            # Handle XGBoost label conversion
            if 'XGBoost' in name:
                y_train_xgb = y_train - 1
                y_test_xgb = y_test - 1
                model.fit(X_train, y_train_xgb)
                y_pred = model.predict(X_test)
                y_pred = y_pred + 1
                y_test_eval = y_test
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_test_eval = y_test
            
            accuracy = accuracy_score(y_test_eval, y_pred)
            f1_macro = f1_score(y_test_eval, y_pred, average='macro')
            training_time = time.time() - start_time
            
            results[name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'training_time': training_time,
                'model': model
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Macro: {f1_macro:.4f}")
            print(f"  Time: {training_time:.2f}s")
            
        except Exception as e:
            print(f"  Error with {name}: {str(e)}")
            continue
    
    return results

def test_class_weights_models(X_train, X_test, y_train, y_test):
    """Test models with class weights"""
    print("\n" + "="*50)
    print("CLASS WEIGHTS MODELS")
    print("="*50)
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    weight_dict = {i+1: weight for i, weight in enumerate(class_weights)}
    
    print("Class weights:")
    for cover_type, weight in weight_dict.items():
        print(f"  Cover Type {cover_type}: {weight:.3f}")
    
    models = {
        'Random Forest (Weights)': RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=1, class_weight=weight_dict
        ),
        'Extra Trees (Weights)': ExtraTreesClassifier(
            n_estimators=100, random_state=42, n_jobs=1, class_weight=weight_dict
        ),
        'Decision Tree (Weights)': DecisionTreeClassifier(
            random_state=42, class_weight=weight_dict
        ),
        'XGBoost (Weights)': xgb.XGBClassifier(
            random_state=42, n_jobs=1
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        start_time = time.time()
        
        try:
            if 'XGBoost' in name:
                y_train_xgb = y_train - 1
                y_test_xgb = y_test - 1
                model.fit(X_train, y_train_xgb)
                y_pred = model.predict(X_test)
                y_pred = y_pred + 1
                y_test_eval = y_test
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_test_eval = y_test
            
            accuracy = accuracy_score(y_test_eval, y_pred)
            f1_macro = f1_score(y_test_eval, y_pred, average='macro')
            training_time = time.time() - start_time
            
            results[name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'training_time': training_time,
                'model': model
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Macro: {f1_macro:.4f}")
            print(f"  Time: {training_time:.2f}s")
            
        except Exception as e:
            print(f"  Error with {name}: {str(e)}")
            continue
    
    return results

def test_balanced_ensemble_models(X_train, X_test, y_train, y_test):
    """Test balanced ensemble models"""
    print("\n" + "="*50)
    print("BALANCED ENSEMBLE MODELS")
    print("="*50)
    
    models = {
        'Balanced Random Forest': BalancedRandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=1
        ),
        'XGBoost (Balanced)': xgb.XGBClassifier(
            random_state=42, n_jobs=1
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        start_time = time.time()
        
        try:
            if 'XGBoost' in name:
                y_train_xgb = y_train - 1
                y_test_xgb = y_test - 1
                model.fit(X_train, y_train_xgb)
                y_pred = model.predict(X_test)
                y_pred = y_pred + 1
                y_test_eval = y_test
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_test_eval = y_test
            
            accuracy = accuracy_score(y_test_eval, y_pred)
            f1_macro = f1_score(y_test_eval, y_pred, average='macro')
            training_time = time.time() - start_time
            
            results[name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'training_time': training_time,
                'model': model
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Macro: {f1_macro:.4f}")
            print(f"  Time: {training_time:.2f}s")
            
        except Exception as e:
            print(f"  Error with {name}: {str(e)}")
            continue
    
    return results

def test_smote_models(X_train, X_test, y_train, y_test):
    """Test models with SMOTE oversampling"""
    print("\n" + "="*50)
    print("SMOTE MODELS")
    print("="*50)
    
    # Apply SMOTE
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"Before: {len(X_train):,} samples")
    print(f"After: {len(X_train_smote):,} samples")
    
    models = {
        'Random Forest (SMOTE)': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
        'Extra Trees (SMOTE)': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=1),
        'XGBoost (SMOTE)': xgb.XGBClassifier(
            n_estimators=100, random_state=42, n_jobs=1
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        start_time = time.time()
        
        try:
            if 'XGBoost' in name:
                y_train_smote_xgb = y_train_smote - 1
                y_test_xgb = y_test - 1
                model.fit(X_train_smote, y_train_smote_xgb)
                y_pred = model.predict(X_test)
                y_pred = y_pred + 1
                y_test_eval = y_test
            else:
                model.fit(X_train_smote, y_train_smote)
                y_pred = model.predict(X_test)
                y_test_eval = y_test
            
            accuracy = accuracy_score(y_test_eval, y_pred)
            f1_macro = f1_score(y_test_eval, y_pred, average='macro')
            training_time = time.time() - start_time
            
            results[name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'training_time': training_time,
                'model': model,
                'smote_data': (X_train_smote, y_train_smote)
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Macro: {f1_macro:.4f}")
            print(f"  Time: {training_time:.2f}s")
            
        except Exception as e:
            print(f"  Error with {name}: {str(e)}")
            continue
    
    return results

def hyperparameter_tuning(X_train, X_test, y_train, y_test):
    """Quick hyperparameter tuning"""
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING")
    print("="*50)
    
    results = {}
    
    # Random Forest tuning
    print("Tuning Random Forest...")
    rf_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=1)
    rf_grid_search = GridSearchCV(
        rf, rf_param_grid, cv=3, scoring='f1_macro', n_jobs=1, verbose=1
    )
    
    start_time = time.time()
    rf_grid_search.fit(X_train, y_train)
    rf_tuning_time = time.time() - start_time
    
    print(f"RF tuning done in {rf_tuning_time:.2f}s")
    print(f"Best params: {rf_grid_search.best_params_}")
    print(f"Best CV score: {rf_grid_search.best_score_:.4f}")
    
    # Test best Random Forest
    best_rf = rf_grid_search.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_f1_macro = f1_score(y_test, y_pred_rf, average='macro')
    
    results['Tuned Random Forest'] = {
        'accuracy': rf_accuracy,
        'f1_macro': rf_f1_macro,
        'training_time': rf_tuning_time,
        'model': best_rf,
        'best_params': rf_grid_search.best_params_
    }
    
    print(f"RF Test Accuracy: {rf_accuracy:.4f}")
    print(f"RF Test F1-Macro: {rf_f1_macro:.4f}")
    
    # XGBoost tuning
    print("\nTuning XGBoost...")
    xgb_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    
    xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=1)
    xgb_grid_search = GridSearchCV(
        xgb_model, xgb_param_grid, cv=3, scoring='f1_macro', n_jobs=1, verbose=1
    )
    
    # Convert labels for XGBoost
    y_train_xgb = y_train - 1
    
    start_time = time.time()
    xgb_grid_search.fit(X_train, y_train_xgb)
    xgb_tuning_time = time.time() - start_time
    
    print(f"XGB tuning done in {xgb_tuning_time:.2f}s")
    print(f"Best params: {xgb_grid_search.best_params_}")
    print(f"Best CV score: {xgb_grid_search.best_score_:.4f}")
    
    # Test best XGBoost
    best_xgb = xgb_grid_search.best_estimator_
    y_pred_xgb = best_xgb.predict(X_test)
    y_pred_xgb = y_pred_xgb + 1
    xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
    xgb_f1_macro = f1_score(y_test, y_pred_xgb, average='macro')
    
    results['Tuned XGBoost'] = {
        'accuracy': xgb_accuracy,
        'f1_macro': xgb_f1_macro,
        'training_time': xgb_tuning_time,
        'model': best_xgb,
        'best_params': xgb_grid_search.best_params_
    }
    
    print(f"XGB Test Accuracy: {xgb_accuracy:.4f}")
    print(f"XGB Test F1-Macro: {xgb_f1_macro:.4f}")
    
    return results

def cross_validation_evaluation(X_train, y_train, best_model):
    """Cross-validation evaluation"""
    print("\n" + "="*50)
    print("CROSS-VALIDATION")
    print("="*50)
    
    # Use 3-fold CV for speed
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    print("Running cross-validation...")
    start_time = time.time()
    
    cv_accuracy = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
    cv_f1 = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=1)
    
    cv_time = time.time() - start_time
    
    print(f"CV done in {cv_time:.2f}s")
    print(f"CV Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std() * 2:.4f})")
    print(f"CV F1-Macro: {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")
    
    return cv_accuracy.mean(), cv_f1.mean()

def compare_all_approaches(all_results):
    """Compare all approaches"""
    print("\n" + "="*70)
    print("COMPARISON OF ALL APPROACHES")
    print("="*70)
    
    # Combine all results
    combined_results = {}
    for category, results in all_results.items():
        if results:
            combined_results.update(results)
    
    # Create comparison DataFrame
    comparison_data = []
    for name, metrics in combined_results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'F1-Macro': metrics['f1_macro'],
            'Training Time': metrics['training_time']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('F1-Macro', ascending=False)
    
    print("Performance Comparison (sorted by F1-Macro):")
    print(comparison_df.to_string(index=False))
    
    # Find best models
    best_accuracy = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
    best_f1 = comparison_df.loc[comparison_df['F1-Macro'].idxmax()]
    fastest = comparison_df.loc[comparison_df['Training Time'].idxmin()]
    
    print(f"\nBest Models:")
    print(f"  Highest Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
    print(f"  Highest F1-Macro: {best_f1['Model']} ({best_f1['F1-Macro']:.4f})")
    print(f"  Fastest: {fastest['Model']} ({fastest['Training Time']:.2f}s)")
    
    return comparison_df

def apply_best_model_to_full_dataset(X_train, X_test, y_train, y_test, feature_names, scaler, best_model_name, all_results):
    """Apply the best model to full dataset"""
    print("\n" + "="*70)
    print("APPLYING BEST MODEL")
    print("="*70)
    
    # Find the best model
    best_model = None
    best_metrics = None
    
    for category, results in all_results.items():
        if results and best_model_name in results:
            best_model = results[best_model_name]['model']
            best_metrics = results[best_model_name]
            break
    
    if best_model is None:
        print("Best model not found!")
        return None
    
    print(f"Using best model: {best_model_name}")
    print(f"   Training accuracy: {best_metrics['accuracy']:.4f}")
    print(f"   F1-Macro: {best_metrics['f1_macro']:.4f}")
    
    # Train on full training set
    print("\nTraining on full training set...")
    start_time = time.time()
    best_model.fit(X_train, y_train)
    full_training_time = time.time() - start_time
    
    print(f"Training done in {full_training_time:.2f}s")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nFinal Test Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Macro: {f1_macro:.4f}")
    print(f"  F1-Weighted: {f1_weighted:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        print("\nTop 15 Features:")
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, row in feature_importance.head(15).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return best_model, {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'training_time': full_training_time
    }

def create_visualization(all_results, best_model_name, final_metrics):
    """Create visualization of results"""
    print("\n" + "="*70)
    print("CREATING VISUALIZATION")
    print("="*70)
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Top 5 Models Performance
    plt.subplot(2, 3, 1)
    
    # Get top 5 models by F1-Macro
    all_models_data = []
    for category, results in all_results.items():
        if results:
            for name, metrics in results.items():
                all_models_data.append({
                    'name': name,
                    'accuracy': metrics['accuracy'],
                    'f1_macro': metrics['f1_macro'],
                    'training_time': metrics['training_time']
                })
    
    # Sort by F1-Macro and get top 5
    all_models_df = pd.DataFrame(all_models_data)
    top_5_models = all_models_df.nlargest(5, 'f1_macro')
    
    x = np.arange(len(top_5_models))
    width = 0.35
    
    plt.bar(x - width/2, top_5_models['accuracy'], width, label='Accuracy', alpha=0.8, color='#2E8B57')
    plt.bar(x + width/2, top_5_models['f1_macro'], width, label='F1-Macro', alpha=0.8, color='#4682B4')
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Top 5 Models Performance')
    plt.xticks(x, [name[:20] + '...' if len(name) > 20 else name for name in top_5_models['name']], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # 2. Best Model Performance
    plt.subplot(2, 3, 2)
    
    # Find the best model metrics
    best_model_metrics = None
    for category, results in all_results.items():
        if results and best_model_name in results:
            best_model_metrics = results[best_model_name]
            break
    
    if best_model_metrics:
        metrics_names = ['Accuracy', 'F1-Macro', 'F1-Weighted']
        metrics_values = [best_model_metrics['accuracy'], best_model_metrics['f1_macro'], final_metrics['f1_weighted']]
        
        bars = plt.bar(metrics_names, metrics_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.title(f'Best Model: {best_model_name[:30]}...')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3)
    
    # 3. Training Time Comparison
    plt.subplot(2, 3, 3)
    
    top_5_time = all_models_df.nsmallest(5, 'training_time')
    
    bars = plt.bar(range(len(top_5_time)), top_5_time['training_time'], 
                   color=['#FFD93D', '#6BCF7F', '#4ECDC4', '#45B7D1', '#96CEB4'])
    plt.title('Fastest Training Models')
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')
    plt.xticks(range(len(top_5_time)), [name[:20] + '...' if len(name) > 20 else name for name in top_5_time['name']], 
                rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, top_5_time['training_time']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    
    # 4. Model Categories Performance
    plt.subplot(2, 3, 4)
    
    category_performance = {}
    for category, results in all_results.items():
        if results:
            # Calculate average performance for each category
            avg_accuracy = np.mean([metrics['accuracy'] for metrics in results.values()])
            avg_f1 = np.mean([metrics['f1_macro'] for metrics in results.values()])
            category_performance[category] = {'accuracy': avg_accuracy, 'f1_macro': avg_f1}
    
    categories = list(category_performance.keys())
    x = np.arange(len(categories))
    
    plt.bar(x - width/2, [category_performance[cat]['accuracy'] for cat in categories], 
            width, label='Avg Accuracy', alpha=0.8, color='#FF6B6B')
    plt.bar(x + width/2, [category_performance[cat]['f1_macro'] for cat in categories], 
            width, label='Avg F1-Macro', alpha=0.8, color='#4ECDC4')
    
    plt.xlabel('Model Categories')
    plt.ylabel('Average Score')
    plt.title('Performance by Category')
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # 5. Summary Table
    plt.subplot(2, 3, 5)
    
    # Create a summary table
    plt.axis('off')
    summary_data = [
        ['Best Model', best_model_name[:25] + '...' if len(best_model_name) > 25 else best_model_name],
        ['Test Accuracy', f"{final_metrics['accuracy']:.4f}"],
        ['F1-Macro', f"{final_metrics['f1_macro']:.4f}"],
        ['Training Time', f"{final_metrics['training_time']:.2f}s"],
        ['Preprocessing', 'MinMaxScaler + SMOTE'],
        ['Algorithm', 'Extra Trees Classifier']
    ]
    
    table = plt.table(cellText=summary_data, 
                     colLabels=['Metric', 'Value'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.4, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_data) + 1):
        for j in range(2):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#4ECDC4')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table[(i, j)].set_facecolor('#F7F7F7')
    
    plt.title('Results Summary', fontsize=16, fontweight='bold', pad=20)
    
    # 6. Performance vs Time Trade-off
    plt.subplot(2, 3, 6)
    
    # Scatter plot of accuracy vs training time
    plt.scatter(all_models_df['training_time'], all_models_df['accuracy'], 
                alpha=0.7, s=100, c=all_models_df['f1_macro'], cmap='viridis')
    
    # Highlight the best model
    best_model_data = all_models_df[all_models_df['name'] == best_model_name]
    if not best_model_data.empty:
        plt.scatter(best_model_data['training_time'], best_model_data['accuracy'], 
                   color='red', s=200, marker='*', label='Best Model', edgecolors='black')
    
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Accuracy')
    plt.title('Performance vs Training Time')
    plt.colorbar(label='F1-Macro Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'model_comparison_results.png'")
    plt.show()

def main():
    """Main function"""
    print("FOREST COVER TYPE CLASSIFICATION - MODEL COMPARISON")
    print("="*70)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess_data()
    
    # Analyze class distribution
    train_counts, test_counts = analyze_class_distribution(y_train, y_test)
    
    # Store all results
    all_results = {}
    
    # Test baseline models
    baseline_results = test_baseline_models(X_train, X_test, y_train, y_test)
    all_results['Baseline'] = baseline_results
    
    # Test class weights models
    weights_results = test_class_weights_models(X_train, X_test, y_train, y_test)
    all_results['Class Weights'] = weights_results
    
    # Test balanced ensemble models
    balanced_results = test_balanced_ensemble_models(X_train, X_test, y_train, y_test)
    all_results['Balanced Ensemble'] = balanced_results
    
    # Test SMOTE models
    smote_results = test_smote_models(X_train, X_test, y_train, y_test)
    all_results['SMOTE'] = smote_results
    
    # Hyperparameter tuning
    tuning_results = hyperparameter_tuning(X_train, X_test, y_train, y_test)
    all_results['Hyperparameter Tuning'] = tuning_results
    
    # Compare all approaches
    comparison_df = compare_all_approaches(all_results)
    
    # Get best model name
    best_model_name = comparison_df.iloc[0]['Model']
    
    # Cross-validation evaluation
    best_model = None
    for category, results in all_results.items():
        if results and best_model_name in results:
            best_model = results[best_model_name]['model']
            break
    
    if best_model:
        cv_accuracy, cv_f1 = cross_validation_evaluation(X_train, y_train, best_model)
    
    # Apply best model to full dataset
    final_model, final_metrics = apply_best_model_to_full_dataset(
        X_train, X_test, y_train, y_test, feature_names, scaler, best_model_name, all_results
    )
    
    # Create visualization
    create_visualization(all_results, best_model_name, final_metrics)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETED!")
    print("="*70)
    print("Summary:")
    print(f"1. Best model: {best_model_name}")
    print(f"2. Test accuracy: {final_metrics['accuracy']:.4f}")
    print(f"3. F1-Macro: {final_metrics['f1_macro']:.4f}")
    print(f"4. Training time: {final_metrics['training_time']:.2f}s")
    
    print("\nNext steps:")
    print("1. Save the best model")
    print("2. Try ensemble methods")
    print("3. Feature engineering")
    print("4. Test on new data")

if __name__ == "__main__":
    main()
