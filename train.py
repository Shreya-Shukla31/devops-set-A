import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText

load_dotenv()

SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASS = os.getenv("SMTP_PASS")
TO_EMAIL = os.getenv("TO_EMAIL")


def send_email(subject, body):
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = SMTP_EMAIL
        msg["To"] = TO_EMAIL

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASS)
        server.sendmail(SMTP_EMAIL, TO_EMAIL, msg.as_string())
        server.quit()

        print("✅ Email sent successfully")

    except Exception as e:
        print("❌ Email failed:", e)


def main():
    try:
        os.makedirs("model", exist_ok=True)

        df = pd.read_csv("data/dataset.csv")

        # last column is target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        mlflow.set_experiment("DevOps_MLOps_Lab")

        with mlflow.start_run():
            model = LogisticRegression(max_iter=200)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # Log parameters + metrics
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("max_iter", 200)
            mlflow.log_metric("accuracy", acc)

            # Save model
            joblib.dump(model, "model/model.pkl")

            # Log model artifact
            mlflow.sklearn.log_model(model, "trained_model")

            print("✅ Training Completed")
            print("Accuracy:", acc)

    except Exception as e:
        error_msg = str(e)
        print("❌ Training Failed:", error_msg)

        # log error into file
        with open("error.log", "a") as f:
            f.write(error_msg + "\n")

        # send email only on failure
        send_email("❌ ALERT: Training Failed", error_msg)


if __name__ == "__main__":
    main()