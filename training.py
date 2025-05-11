import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

def preprocess_data(X_train, X_test):
    # 분산이 낮은 특성 제거
    selector = VarianceThreshold(threshold=0.01)
    X_train = selector.fit_transform(X_train)
    X_test = selector.transform(X_test)

    # 표준화
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    logging.info("모델 초기화 및 학습 시작")
    
    # 모델 구성
    model = Sequential([
        Dense(128, input_dim=X_train.shape[1], activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(len(set(y_train)), activation='softmax')
    ])
    
    # 모델 컴파일
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 모델 학습
    logging.info("모델 학습 진행")
    model.fit(
        X_train, 
        y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        verbose=1
    )
    
    # 예측
    logging.info("예측 시작")
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    
    # 평가
    logging.info("모델 평가 시작")
    accuracy = accuracy_score(y_test, y_pred_classes)
    report = classification_report(y_test, y_pred_classes)
    
    return model, accuracy, report