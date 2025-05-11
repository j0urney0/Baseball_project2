import logging
from data_preprocessing import load_and_preprocess_data
from feature_engineering import feature_engineering
from training import train_and_evaluate_model, preprocess_data
from visualization import plot_confusion_matrix

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("데이터 로드 및 전처리 시작")
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data()

    logging.info("특성 엔지니어링 시작")
    X_train = feature_engineering(X_train)
    X_test = feature_engineering(X_test)

    logging.info("데이터 전처리 시작")
    X_train, X_test = preprocess_data(X_train, X_test)

    logging.info("모델 학습 및 평가 시작")
    model, accuracy, report = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    logging.info(f"모델 정확도: {accuracy}")
    logging.info(f"분류 보고서:\n{report}")

    logging.info("결과 시각화 시작")
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    plot_confusion_matrix(y_test, y_pred_classes)

if __name__ == "__main__":
    main()